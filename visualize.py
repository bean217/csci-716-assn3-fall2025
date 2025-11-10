"""
Visualization utilities for trapezoidal maps.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, FancyBboxPatch
from collections import deque
from typing import Dict, List, Tuple
import utils
from utils import Node, XNode, YNode, Leaf


def visualize_trapezoidal_map(trap_map, title: str = "Trapezoidal Map"):
    """Visualize trapezoidal map using matplotlib."""
    _, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

    bbox = trap_map.bbox
    padding = 2.0
    ax.set_xlim(bbox[0] - padding, bbox[2] + padding)
    ax.set_ylim(bbox[1] - padding, bbox[3] + padding)

    # Draw bounding box
    from matplotlib.patches import Rectangle
    bbox_rect = Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
    )
    ax.add_patch(bbox_rect)

    # Collect unique segments
    segments_set = set()
    for trap in trap_map.trapezoids:
        if trap.top:
            segments_set.add(trap.top)
        if trap.bottom:
            segments_set.add(trap.bottom)

    # Draw trapezoids
    colors = plt.cm.Set3(np.linspace(0, 1, len(trap_map.trapezoids)))
    for i, trap in enumerate(trap_map.trapezoids):
        xl, xr = trap.leftx, trap.rightx
        y_top_left = trap.y_top(xl, bbox[3])
        y_top_right = trap.y_top(xr, bbox[3])
        y_bot_left = trap.y_bottom(xl, bbox[1])
        y_bot_right = trap.y_bottom(xr, bbox[1])

        # Build trapezoid vertices (counter-clockwise)
        vertices = [
            [xl, y_bot_left],
            [xr, y_bot_right],
            [xr, y_top_right],
            [xl, y_top_left]
        ]

        poly = Polygon(vertices, alpha=0.3, facecolor=colors[i],
                      edgecolor='gray', linewidth=0.5)
        ax.add_patch(poly)

        # Draw vertical boundaries
        ax.plot([xl, xl], [y_bot_left, y_top_left], 'k-', linewidth=1, alpha=0.5)
        ax.plot([xr, xr], [y_bot_right, y_top_right], 'k-', linewidth=1, alpha=0.5)

        # Label trapezoid
        if trap.label:
            center_x = (xl + xr) / 2
            center_y = (y_top_left + y_top_right + y_bot_left + y_bot_right) / 4
            ax.text(center_x, center_y, trap.label,
                   fontsize=10, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Draw segments
    for seg in segments_set:
        ax.plot([seg.left.x, seg.right.x], [seg.left.y, seg.right.y],
               'b-', linewidth=2, marker='o', markersize=5)
        ax.text(seg.left.x, seg.left.y, f'  {seg.left.label}',
               fontsize=8, ha='left', va='bottom')
        ax.text(seg.right.x, seg.right.y, f'  {seg.right.label}',
               fontsize=8, ha='left', va='bottom')

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.text(0.02, 0.98, f'Trapezoids: {len(trap_map.trapezoids)}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def visualize_dag(trap_map, title: str = "Trapezoidal Map DAG"):
    """Visualize the DAG structure of the trapezoidal map using BFS layout."""
    if not trap_map.root:
        print("No DAG to visualize (empty map)")
        return

    # BFS to assign levels and positions to nodes
    # Allow leaf nodes to appear multiple times to create tree structure
    levels: Dict[int, List[Tuple[Node, int]]] = {}  # level -> list of (node, unique_id)
    node_positions: Dict[int, Tuple[float, float]] = {}  # unique_id -> (x, y)
    node_id_counter = [0]  # Use list to allow modification in nested scope

    def get_next_id():
        node_id_counter[0] += 1
        return node_id_counter[0]

    queue = deque([(trap_map.root, 0, get_next_id())])  # (node, level, unique_id)
    visited = {id(trap_map.root)}  # Only track non-leaf nodes to avoid cycles

    while queue:
        node, level, unique_id = queue.popleft()

        if level not in levels:
            levels[level] = []
        levels[level].append((node, unique_id))

        # Add children to queue
        children = []
        if isinstance(node, XNode):
            if node.left:
                children.append(node.left)
            if node.right:
                children.append(node.right)
        elif isinstance(node, YNode):
            if node.above:
                children.append(node.above)
            if node.below:
                children.append(node.below)

        for child in children:
            child_unique_id = get_next_id()
            # Only mark non-leaf nodes as visited to allow leaves to appear multiple times
            if not isinstance(child, Leaf):
                if id(child) in visited:
                    continue
                visited.add(id(child))
            queue.append((child, level + 1, child_unique_id))

    # Calculate positions for each node
    max_width = max(len(nodes) for nodes in levels.values())
    level_height = 2.0
    node_spacing = 3.0

    for level, node_list in levels.items():
        y = -level * level_height
        width = len(node_list)
        start_x = -(width - 1) * node_spacing / 2

        for i, (node, unique_id) in enumerate(node_list):
            x = start_x + i * node_spacing
            node_positions[unique_id] = (x, y)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, max_width * 2), max(8, len(levels) * 2)))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Helper function to get node label
    def get_label(node: Node) -> str:
        if isinstance(node, XNode):
            return node.point.label if node.point.label else "P?"
        elif isinstance(node, YNode):
            return node.seg.label if node.seg.label else "S?"
        elif isinstance(node, Leaf):
            return node.trap.label if node.trap.label else "T?"
        return "?"

    # Helper function to get node color
    def get_color(node: Node) -> Tuple[str, str]:
        """Return (fill_color, border_color) for node type."""
        if isinstance(node, XNode):
            return ('#ADD8E6', '#0000FF')  # LightBlue/Blue for point nodes
        elif isinstance(node, YNode):
            return ('#FFD580', '#FF8C00')  # Light orange/DarkOrange for segment nodes
        elif isinstance(node, Leaf):
            return ('#E6E6FA', '#4B0082')  # Lavender/Indigo for leaf nodes
        return ('#D3D3D3', '#696969')     # LightGray/DimGray for unknown

    # Create a mapping to track children for each unique_id
    node_children_map: Dict[int, List[Tuple[str, int]]] = {}  # unique_id -> [(edge_label, child_unique_id)]

    # Build parent-child relationships with unique IDs
    for level, node_list in levels.items():
        for node, unique_id in node_list:
            children = []
            if isinstance(node, XNode):
                if node.left:
                    children.append(('Left', node.left))
                if node.right:
                    children.append(('Right', node.right))
            elif isinstance(node, YNode):
                if node.above:
                    children.append(('Above', node.above))
                if node.below:
                    children.append(('Below', node.below))

            # Find the unique IDs of children (they should be in the next level)
            if children and level + 1 in levels:
                node_children_map[unique_id] = []
                next_level = levels[level + 1]
                child_idx = 0
                for edge_label, child_node in children:
                    # Find this child in the next level
                    for child_node_obj, child_unique_id in next_level:
                        if child_node is child_node_obj and child_unique_id not in [c[1] for c in node_children_map[unique_id]]:
                            node_children_map[unique_id].append((edge_label, child_unique_id))
                            break

    # Draw edges first (so they appear behind nodes)
    for parent_unique_id, children in node_children_map.items():
        x, y = node_positions[parent_unique_id]
        for edge_label, child_unique_id in children:
            cx, cy = node_positions[child_unique_id]
            # Draw edge
            ax.plot([x, cx], [y, cy], 'k-', linewidth=1.5, alpha=0.6, zorder=1)

            # Add edge label
            mid_x, mid_y = (x + cx) / 2, (y + cy) / 2
            ax.text(mid_x, mid_y, edge_label, fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.1', facecolor='white',
                            edgecolor='none', alpha=0.8), zorder=2)

    # Draw nodes
    for level_nodes in levels.values():
        for node, unique_id in level_nodes:
            x, y = node_positions[unique_id]
            label = get_label(node)
            fill_color, border_color = get_color(node)

            if isinstance(node, Leaf):
                # Leaves are squares
                size = 0.6
                rect = FancyBboxPatch((x - size/2, y - size/2), size, size,
                                     boxstyle='round,pad=0.05',
                                     facecolor=fill_color, edgecolor=border_color,
                                     linewidth=2, zorder=3)
                ax.add_patch(rect)
            else:
                # XNodes and YNodes are circles
                circle = plt.Circle((x, y), 0.35, facecolor=fill_color, edgecolor=border_color,
                                   linewidth=2, zorder=3)
                ax.add_patch(circle)

            # Add label text
            ax.text(x, y, label, fontsize=10, ha='center', va='center',
                   fontweight='bold', zorder=4)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ADD8E6',
                   markeredgecolor='#0000FF', markersize=10, label='XNode (Point)',
                   markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD580',
                   markeredgecolor='#FF8C00', markersize=10, label='YNode (Segment)',
                   markeredgewidth=2),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#E6E6FA',
                   markeredgecolor='#4B0082', markersize=10, label='Leaf (Trapezoid)',
                   markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add info text
    total_nodes = sum(len(nodes) for nodes in levels.values())
    ax.text(0.02, 0.98, f'Total nodes: {total_nodes}\nLevels: {len(levels)}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Set axis limits with padding
    all_x = [pos[0] for pos in node_positions.values()]
    all_y = [pos[1] for pos in node_positions.values()]
    padding = 1.0
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)

    plt.tight_layout()
    plt.show()
