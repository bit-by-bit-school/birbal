# This module extracts data from org roam and parses it into a data frame

import os
import glob
import orgparse
import pandas as pd
import re

def get_all_filenames_in_roam():
    roam_path = os.getenv('ROAM_PATH')
    path = os.path.join(roam_path, "**/*.org")
    files = glob.glob(path, recursive=True)

    return files

def extract_title(node):
    if node.heading:
        return node.heading

    title_pattern = re.compile(r'^#\+title:\s*(.*)$', re.IGNORECASE)
    match = title_pattern.search(node.body)
    if match:
        return match.group(1)
    else:
        return re.sub(r"#\+title:",
		"",
		node.body.split("\n")[0], flags=re.IGNORECASE).strip()

def extract_node_nested_body(node):
    body = node.body
    for child in node.children:
        body += '\n' + child.level * "*" + " " + child.heading + "\n" + \
		extract_node_nested_body(child)
    return body.strip()

def extract_node_nested_body_exclusive(node):
    body = node.body
    for child in node.children:
        if not child.properties.get('ID') and not child.properties.get('SEARCH'):
            body += '\n' + child.level * "*" + " " + child.heading + "\n" + \
			extract_node_nested_body_exclusive(child)
    return body.strip()

def build_node_hierarchy(node):
    hierarchy = [extract_title(node)]
    parent = node.parent
    # while parent and parent != org_data[0]:
    while parent:
        hierarchy.append(extract_title(parent))
        parent = parent.parent
    return ' > '.join(reversed(hierarchy)).strip()

def node_to_dict(node, file_name):
    node_dict = {
        'file_name': file_name,
        'node_id': node.properties.get('ID'),
        'node_title': extract_title(node),
        'node_hierarchy': build_node_hierarchy(node),
        'node_text': node.body,
        'node_text_nested': extract_node_nested_body(node),
        'node_text_nested_exclusive': extract_node_nested_body_exclusive(node),
    }
    return node_dict

def org_roam_nodes_to_dataframe(org_file):
    # Load the org file into an OrgData object
    org_data = orgparse.load(org_file)
    # Create a list of all org-roam nodes in the OrgData object
    nodes = [node_to_dict(node, org_file) for node in org_data[0][:] if node.properties.get('ID')]

    return pd.DataFrame(nodes)

def org_files_to_dataframes():
    roam_nodes_df =  pd.concat([org_roam_nodes_to_dataframe(file) for file in get_all_filenames_in_roam()])
    roam_nodes_df["text_to_encode"] = (
        roam_nodes_df["node_text_nested_exclusive"]
        .astype(str)
        .str.replace("#+filetags:", "tags:")
        .str.replace("#+title:", "title:")
    )
    roam_nodes_df["text_to_encode"] = (
        "[" + roam_nodes_df["node_hierarchy"] + "] " +
        roam_nodes_df["text_to_encode"].astype(str)
    )
    return roam_nodes_df
