"""SVG parsing, element extraction, and subpath splitting.

Parses SVG XML into structured elements for LOO analysis. Supports two
granularity levels:

- **Element-level** (``parse_svg``): each top-level visual element (path,
  circle, rect, ...) or flattened group child becomes one unit.
- **Subpath-level** (``parse_svg_subpaths``): compound ``<path>`` elements
  with multiple M/m commands are split at boundaries, giving finer-grained
  attribution.

Adapted from ``incremental_path_scorer.py`` and ``subpath_scorer.py`` in the
SVG-Evaluation-Metrics research codebase.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

from lxml import etree

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"
STRUCTURAL_TAGS = frozenset({"defs", "style", "metadata", "title", "desc"})
VISUAL_TAGS = frozenset({
    "path", "circle", "rect", "line", "polygon", "polyline",
    "ellipse", "text", "use", "image", "g",
})


@dataclass
class SVGElement:
    """A visual element extracted from an SVG."""

    index: int
    tag: str  # e.g. 'path', 'circle', 'g', 'subpath'
    xml: str  # serialized XML fragment
    num_children: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _localname(tag: object) -> str:
    """Get local name from possibly namespaced tag."""
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return tag if isinstance(tag, str) else ""


def _element_summary(elem: etree._Element) -> str:
    """Create a human-readable summary of an SVG element."""
    tag = _localname(elem.tag)
    if tag == "path":
        d = elem.get("d", "")
        return f"path(d={d[:60]}{'...' if len(d) > 60 else ''})"
    elif tag == "circle":
        return f"circle(cx={elem.get('cx')}, cy={elem.get('cy')}, r={elem.get('r')})"
    elif tag == "rect":
        return (
            f"rect(x={elem.get('x', 0)}, y={elem.get('y', 0)}, "
            f"w={elem.get('width')}, h={elem.get('height')})"
        )
    elif tag == "ellipse":
        return (
            f"ellipse(cx={elem.get('cx')}, cy={elem.get('cy')}, "
            f"rx={elem.get('rx')}, ry={elem.get('ry')})"
        )
    elif tag == "line":
        return (
            f"line({elem.get('x1')},{elem.get('y1')} -> "
            f"{elem.get('x2')},{elem.get('y2')})"
        )
    elif tag == "g":
        n = len(list(elem))
        return f"g({n} children)"
    elif tag == "text":
        txt = (elem.text or "")[:30]
        return f"text('{txt}')"
    elif tag == "polygon":
        pts = elem.get("points", "")
        return f"polygon(points={pts[:40]}...)"
    elif tag == "polyline":
        pts = elem.get("points", "")
        return f"polyline(points={pts[:40]}...)"
    else:
        return f"{tag}()"


# ---------------------------------------------------------------------------
# Element-level parsing
# ---------------------------------------------------------------------------


def parse_svg(
    svg_content: str, flatten_groups: bool = True,
) -> Tuple[dict, list, List[SVGElement], dict]:
    """Parse SVG into root attributes, structural elements, and visual elements.

    Parameters
    ----------
    svg_content : str
        SVG XML string.
    flatten_groups : bool
        If *True*, recursively flatten ``<g>`` groups into individual elements.
        If *False*, treat each top-level ``<g>`` as a single element.

    Returns
    -------
    tuple of (root_attribs, structural_elements, visual_elements, nsmap)
    """
    root = etree.fromstring(
        svg_content.encode("utf-8") if isinstance(svg_content, str) else svg_content,
    )

    root_attribs = dict(root.attrib)
    nsmap = root.nsmap

    structural: list = []
    visual_elems: List[SVGElement] = []
    idx = 0

    def _extract_visual(elem: etree._Element) -> None:
        nonlocal idx
        tag = _localname(elem.tag)

        if tag in STRUCTURAL_TAGS:
            return

        if tag == "g" and flatten_groups:
            children = [c for c in elem if isinstance(c.tag, str)]
            visual_children = [
                c for c in children if _localname(c.tag) not in STRUCTURAL_TAGS
            ]
            if visual_children:
                group_transform = elem.get("transform")
                group_style = elem.get("style")
                group_fill = elem.get("fill")
                group_opacity = elem.get("opacity")

                for child in visual_children:
                    child_tag = _localname(child.tag)
                    if child_tag == "g":
                        _extract_visual(child)
                    else:
                        if group_transform or group_style or group_fill or group_opacity:
                            wrapper = deepcopy(elem)
                            for c in list(wrapper):
                                wrapper.remove(c)
                            wrapper.append(deepcopy(child))
                            xml_str = etree.tostring(wrapper, encoding="unicode")
                        else:
                            xml_str = etree.tostring(child, encoding="unicode")

                        visual_elems.append(SVGElement(
                            index=idx,
                            tag=child_tag,
                            xml=xml_str,
                            summary=_element_summary(child),
                        ))
                        idx += 1
            return

        # Non-group element or group treated as single unit
        xml_str = etree.tostring(elem, encoding="unicode")
        n_children = 0
        if tag == "g":
            n_children = len([
                c for c in elem if _localname(c.tag) not in STRUCTURAL_TAGS
            ])

        visual_elems.append(SVGElement(
            index=idx,
            tag=tag,
            xml=xml_str,
            num_children=n_children,
            summary=_element_summary(elem),
        ))
        idx += 1

    for child in root:
        if not isinstance(child.tag, str):
            continue
        tag = _localname(child.tag)
        if tag in STRUCTURAL_TAGS:
            structural.append(child)
        else:
            _extract_visual(child)

    return root_attribs, structural, visual_elems, nsmap


def build_partial_svg(
    root_attribs: dict,
    structural: list,
    elements: List[SVGElement],
    nsmap: dict | None = None,
) -> str:
    """Reconstruct an SVG string from root attributes, structural elements, and a list of visual elements."""
    if nsmap is None:
        nsmap = {None: SVG_NS}
    if None not in nsmap:
        nsmap[None] = SVG_NS

    root = etree.Element(f"{{{SVG_NS}}}svg", nsmap=nsmap)

    for k, v in root_attribs.items():
        if k.startswith("{") and "xmlns" in k:
            continue
        if k == "xmlns" or k.startswith("xmlns:"):
            continue
        root.set(k, v)

    for elem in structural:
        root.append(deepcopy(elem))

    for svg_elem in elements:
        parsed = etree.fromstring(
            svg_elem.xml.encode("utf-8") if isinstance(svg_elem.xml, str) else svg_elem.xml,
        )
        root.append(parsed)

    return etree.tostring(root, encoding="unicode", pretty_print=True)


# ---------------------------------------------------------------------------
# Subpath splitting
# ---------------------------------------------------------------------------


def split_path_d(d: str) -> List[str]:
    """Split a path ``d`` attribute into subpath segments at M/m boundaries.

    Each subpath starts with an M/m command and runs until the next M/m or
    end of string. Returns a list of d-attribute strings, one per subpath.
    If the path has only one subpath, returns a single-element list.
    """
    if not d or not d.strip():
        return []

    m_positions = [m.start() for m in re.finditer(r"[Mm]", d)]

    if len(m_positions) <= 1:
        return [d.strip()]

    subpaths = []
    for i, pos in enumerate(m_positions):
        end = m_positions[i + 1] if i + 1 < len(m_positions) else len(d)
        segment = d[pos:end].strip().rstrip(" ,")
        if segment:
            subpaths.append(segment)

    return subpaths


def parse_svg_subpaths(svg_content: str) -> Tuple[dict, list, List[SVGElement], dict]:
    """Parse SVG with ``<path>`` elements split into subpath-level elements.

    Non-path elements are kept as-is. Path elements with multiple subpaths
    are expanded into one :class:`SVGElement` per subpath, each inheriting
    the parent path's attributes (fill, stroke, transform, etc.).

    Returns the same 4-tuple as :func:`parse_svg`.
    """
    root = etree.fromstring(
        svg_content.encode("utf-8") if isinstance(svg_content, str) else svg_content,
    )

    root_attribs = dict(root.attrib)
    nsmap = root.nsmap
    structural: list = []
    visual_elems: List[SVGElement] = []
    idx = 0

    def _extract(elem: etree._Element) -> None:
        nonlocal idx
        tag = _localname(elem.tag)

        if tag in STRUCTURAL_TAGS:
            return
        if tag == "svg":
            for child in elem:
                if isinstance(child.tag, str):
                    _extract(child)
            return

        if tag == "g":
            children = [c for c in elem if isinstance(c.tag, str)]
            visual_children = [
                c for c in children if _localname(c.tag) not in STRUCTURAL_TAGS
            ]
            if visual_children:
                group_transform = elem.get("transform")
                group_style = elem.get("style")
                group_fill = elem.get("fill")
                group_opacity = elem.get("opacity")
                has_group_attrs = any([
                    group_transform, group_style, group_fill, group_opacity,
                ])
                for child in visual_children:
                    child_tag = _localname(child.tag)
                    if child_tag == "g":
                        _extract(child)
                    elif child_tag == "path":
                        _split_path(
                            child,
                            wrapper_elem=elem if has_group_attrs else None,
                        )
                    else:
                        if has_group_attrs:
                            wrapper = deepcopy(elem)
                            for c in list(wrapper):
                                wrapper.remove(c)
                            wrapper.append(deepcopy(child))
                            xml_str = etree.tostring(wrapper, encoding="unicode")
                        else:
                            xml_str = etree.tostring(child, encoding="unicode")
                        visual_elems.append(SVGElement(
                            index=idx,
                            tag=child_tag,
                            xml=xml_str,
                            summary=_element_summary(child),
                        ))
                        idx += 1
            return

        if tag == "path":
            _split_path(elem)
        else:
            xml_str = etree.tostring(elem, encoding="unicode")
            visual_elems.append(SVGElement(
                index=idx,
                tag=tag,
                xml=xml_str,
                summary=_element_summary(elem),
            ))
            idx += 1

    def _split_path(
        path_elem: etree._Element,
        wrapper_elem: etree._Element | None = None,
    ) -> None:
        nonlocal idx
        d = path_elem.get("d", "")
        subpaths = split_path_d(d)

        if len(subpaths) <= 1:
            if wrapper_elem is not None:
                wrapper = deepcopy(wrapper_elem)
                for c in list(wrapper):
                    wrapper.remove(c)
                wrapper.append(deepcopy(path_elem))
                xml_str = etree.tostring(wrapper, encoding="unicode")
            else:
                xml_str = etree.tostring(path_elem, encoding="unicode")

            d_preview = d[:60] + ("..." if len(d) > 60 else "")
            visual_elems.append(SVGElement(
                index=idx,
                tag="path",
                xml=xml_str,
                summary=f"path(d={d_preview})",
            ))
            idx += 1
            return

        for si, subpath_d in enumerate(subpaths):
            new_path = deepcopy(path_elem)
            new_path.set("d", subpath_d)

            if wrapper_elem is not None:
                wrapper = deepcopy(wrapper_elem)
                for c in list(wrapper):
                    wrapper.remove(c)
                wrapper.append(new_path)
                xml_str = etree.tostring(wrapper, encoding="unicode")
            else:
                xml_str = etree.tostring(new_path, encoding="unicode")

            n_cmds = len(re.findall(r"[A-Za-z]", subpath_d))
            d_preview = subpath_d[:50] + ("..." if len(subpath_d) > 50 else "")
            visual_elems.append(SVGElement(
                index=idx,
                tag="subpath",
                xml=xml_str,
                summary=f"subpath[{si}/{len(subpaths)}](d={d_preview}, {n_cmds} cmds)",
            ))
            idx += 1

    for child in root:
        if not isinstance(child.tag, str):
            continue
        tag = _localname(child.tag)
        if tag in STRUCTURAL_TAGS:
            structural.append(child)
        else:
            _extract(child)

    return root_attribs, structural, visual_elems, nsmap
