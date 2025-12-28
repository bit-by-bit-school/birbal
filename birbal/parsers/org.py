# This module extracts data from org roam and parses it into a data frame
import os
import glob
import orgparse
import pandas as pd
import re
from pathlib import Path
from birbal.config import config


class OrgParser(DocumentParser):
    # ---------- PRIVATE HELPERS ----------
    def _extract_title(self, node):
        if node.heading:
            return node.heading

        match = re.search(r"^#\+title:\s*(.*)$", node.body, re.IGNORECASE)
        if match:
            return match.group(1)

        return re.sub(
            r"#\+title:", "", node.body.split("\n")[0], flags=re.IGNORECASE
        ).strip()

    def _extract_node_nested_body(self, node):
        body = node.body
        for child in node.children:
            body += (
                "\n"
                + child.level * "*"
                + " "
                + child.heading
                + "\n"
                + self._extract_node_nested_body(child)
            )
        return body.strip()

    def _format_org_roam_links(self, node_body):
        ORG_ROAM_LINK_RE = re.compile(r"\[\[id:([^\]]+)\]\[([^\]]+)\]\]")

        def repl(match):
            title = match.group(2)
            return f"{title} [RELATED NOTE: {title}]"

        return ORG_ROAM_LINK_RE.sub(repl, node_body)

    def _extract_node_nested_body_exclusive(self, node):
        body = node.get_body(format="raw")
        for child in node.children:
            if not child.properties.get("ID") and not child.properties.get("SEARCH"):
                body += (
                    "\n"
                    + child.level * "*"
                    + " "
                    + child.heading
                    + "\n"
                    + self._extract_node_nested_body_exclusive(child)
                )

        body = body.replace("#+filetags:", "tags:").replace("#+title:", "title:")
        return self._format_org_roam_links(body).strip()

    def _build_node_hierarchy(self, node):
        hierarchy = [self._extract_title(node)]
        parent = node.parent
        while parent:
            hierarchy.append(self._extract_title(parent))
            parent = parent.parent
        return hierarchy

    def _node_to_dict(self, node, file_name):
        return {
            "file_name": file_name,
            "root_id": node.properties.get("ID"),
            "title": self._extract_title(node),
            "hierarchy": self._build_node_hierarchy(node),
            "text": self._extract_node_nested_body_exclusive(node),
        }

    def _split_node_by_org_headings(self, node_dict):
        root_text = node_dict["text"]
        base_hierarchy = node_dict["hierarchy"]
        base_id = node_dict["root_id"]

        def split_recursive(text, depth, parent_titles):
            star_pattern = rf"\n\*{{{depth}}}\s+"
            parts = re.split(star_pattern, text)

            if len(parts) == 1:
                return [{**node_dict, "text": text, "hierarchy": parent_titles}]

            children = []
            for part in parts:
                lines = part.splitlines()
                title = lines[0].strip()

                new_parents = (
                    parent_titles
                    if title.lower().startswith("title:")
                    else parent_titles + [title]
                )

                children.extend(split_recursive(part, depth + 1, new_parents))
            return children

        split_nodes = split_recursive(root_text, 1, base_hierarchy)
        for i, node in enumerate(split_nodes):
            node["id"] = f"{base_id}.{i}"
        return split_nodes

    def _format_node(self, node_dict):
        formatted_hierarchy = " > ".join(reversed(node_dict["hierarchy"])).strip()
        return {
            **node_dict,
            "hierarchy": formatted_hierarchy,
            "text": f"[{formatted_hierarchy}] {node_dict['text']}",
        }

    # ---------- PUBLIC API ----------
    def parse(self, path: Path) -> pd.DataFrame:
        org_data = orgparse.load(path)

        nodes = [
            self._node_to_dict(node, path)
            for node in org_data[0][:]
            if node.properties.get("ID")
        ]

        split_nodes = [
            sub
            for node_dict in nodes
            for sub in self._split_node_by_org_headings(node_dict)
        ]

        formatted = [self._format_node(node) for node in split_nodes]
        return pd.DataFrame(formatted)


