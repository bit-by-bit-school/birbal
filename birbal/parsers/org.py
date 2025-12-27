# This module extracts data from org roam and parses it into a data frame

import os
import glob
import orgparse
import pandas as pd
import re
from config import config


def get_all_filenames_in_roam():
    org_path = config["file_dir"]
    path = os.path.join(org_path, "**/*.org")
    files = glob.glob(path, recursive=True)

    return files


def extract_title(node):
    if node.heading:
        return node.heading

    title_pattern = re.compile(r"^#\+title:\s*(.*)$", re.IGNORECASE)
    match = title_pattern.search(node.body)
    if match:
        return match.group(1)
    else:
        return re.sub(
            r"#\+title:", "", node.body.split("\n")[0], flags=re.IGNORECASE
        ).strip()


def extract_node_nested_body(node):
    body = node.body
    for child in node.children:
        body += (
            "\n"
            + child.level * "*"
            + " "
            + child.heading
            + "\n"
            + extract_node_nested_body(child)
        )

    return body.strip()


def format_org_roam_links(node_body):
    """Note that this uses the descriptive title as the related note,
    which may not be the actual note name -> possible to improve?"""
    ORG_ROAM_LINK_RE = re.compile(r"\[\[id:([^\]]+)\]\[([^\]]+)\]\]")

    def rewrite_roam_link(match):
        title = match.group(2)
        return f"{title} [RELATED NOTE: {title}]"

    return ORG_ROAM_LINK_RE.sub(rewrite_roam_link, node_body)


def extract_node_nested_body_exclusive(node):
    body = node.get_body(format="raw")

    for child in node.children:
        if not child.properties.get("ID") and not child.properties.get("SEARCH"):
            body += (
                "\n"
                + child.level * "*"
                + " "
                + child.heading
                + "\n"
                + extract_node_nested_body_exclusive(child)
            )

    body = body.replace("#+filetags:", "tags:").replace("#+title:", "title:")
    body = format_org_roam_links(body)

    return body.strip()


def build_node_hierarchy(node):
    hierarchy = [extract_title(node)]
    parent = node.parent
    # while parent and parent != org_data[0]:
    while parent:
        hierarchy.append(extract_title(parent))
        parent = parent.parent

    return hierarchy


def node_to_dict(node, file_name):
    node_dict = {
        "file_name": file_name,
        "root_id": node.properties.get("ID"),
        "title": extract_title(node),
        "hierarchy": build_node_hierarchy(node),
        "text": extract_node_nested_body_exclusive(node),
    }
    return node_dict


def split_node_by_org_headings(node_dict):
    root_text = node_dict["text"]
    base_hierarchy = node_dict["hierarchy"]
    base_id = node_dict["root_id"]

    def split_recursive(text, depth, parent_titles):
        star_pattern = r"\n\*{" + str(depth) + r"}\s+"
        parts = re.split(star_pattern, text)

        # If no further splits possible, this is a leaf node
        if len(parts) == 1:
            return [
                {
                    **node_dict,
                    "text": text,
                    "hierarchy": parent_titles,
                }
            ]

        children = []
        for part in parts:
            lines = part.splitlines()
            title = lines[0].strip()

            extended_parents = (
                parent_titles
                if title.lower().startswith("title:")
                else parent_titles + [title]
            )

            children.extend(split_recursive(part, depth + 1, extended_parents))
        return children

    split_nodes = split_recursive(root_text, 1, base_hierarchy)
    for i, node in enumerate(split_nodes):
        node["id"] = f"{base_id}.{i}"

    return split_nodes


def format_node(node_dict):
    formatted_hierarchy = f" > ".join(reversed(node_dict["hierarchy"])).strip()
    text_to_encode = "[" + formatted_hierarchy + "] " + node_dict["text"]

    return {
        **node_dict,
        "hierarchy": formatted_hierarchy,
        "text": text_to_encode,
    }


def org_roam_nodes_to_dataframe(org_file):
    org_data = orgparse.load(org_file)
    nodes = [
        node_to_dict(node, org_file)
        for node in org_data[0][:]
        if node.properties.get("ID")
    ]
    split_nodes = [
        subnode
        for node_dict in nodes
        for subnode in split_node_by_org_headings(node_dict)
    ]
    formatted_nodes = [format_node(node) for node in split_nodes]

    return pd.DataFrame(formatted_nodes)


def org_files_to_dataframes():
    roam_nodes_df = pd.concat(
        [org_roam_nodes_to_dataframe(file) for file in get_all_filenames_in_roam()]
    )
    return roam_nodes_df
