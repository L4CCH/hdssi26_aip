"""
app.py

Locally hosted interactive knowledge graph for consolidated
GLiNER2 relation-extraction results.

Expected local file:
- gliner2_relations_consolidated.json

Run locally:
    streamlit run app.py \
        --server.address localhost \
        --server.port 8000

Then open:
    http://localhost:8000
"""

import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


OUTPUT_FILE = "gliner2_relations_consolidated.json"

PAGE_TITLE = (
    "AIP Oral History Transcripts: "
    "Relation Knowledge Graph"
)

GRAPH_HEIGHT = 440


st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


# light-mode styling
st.markdown(
    """
    <style>
        :root {
            color-scheme: light;
        }

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background-color: #ffffff !important;
            color: #111111 !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
            height: 2rem;
            border: none;
        }

        [data-testid="stSidebar"] {
            background-color: #f7f7f7 !important;
            border-right: 1px solid #d9d9d9;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: #111111 !important;
        }

        .block-container {
            max-width: 100%;
            padding-top: 1rem;
            padding-right: 1.5rem;
            padding-bottom: 2rem;
            padding-left: 1.5rem;
        }

        [data-testid="stMetric"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0.1rem 0.5rem 0.5rem 0.5rem;
        }

        [data-testid="stMetricLabel"] {
            color: #555555 !important;
        }

        [data-testid="stMetricValue"] {
            color: #111111 !important;
        }

        [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #111111 !important;
        }

        [data-baseweb="popover"],
        [data-baseweb="menu"] {
            background-color: #ffffff !important;
            color: #111111 !important;
        }

        h1,
        h2,
        h3 {
            color: #111111 !important;
        }

        h2,
        h3 {
            margin-top: 0.75rem !important;
            margin-bottom: 0.5rem !important;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_relations(path: str) -> list[dict]:
    """
    Load and validate the consolidated relation JSON file.
    """

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Expected the JSON file to contain a list "
            "of relation objects."
        )

    required_fields = {
        "head",
        "tail",
        "relation",
        "total_count",
    }

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(
                f"Relation at index {index} is not an object."
            )

        missing_fields = required_fields - record.keys()

        if missing_fields:
            raise ValueError(
                f"Relation at index {index} is missing: "
                f"{sorted(missing_fields)}"
            )

    return data


def safe_integer(value, default: int = 0) -> int:
    """
    Convert a value to an integer without crashing.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def calculate_entity_metrics(
    relations: list[dict],
) -> tuple[Counter, Counter]:
    """
    Calculate entity frequency and connectivity.

    Frequency:
        Sum of total_count for every relation involving
        the entity.

    Connectivity:
        Number of unique entities connected to the entity.
    """

    entity_frequency = Counter()
    entity_neighbors = defaultdict(set)

    for record in relations:
        head = str(record.get("head", "")).strip()
        tail = str(record.get("tail", "")).strip()

        if not head or not tail:
            continue

        total_count = max(
            safe_integer(
                record.get("total_count"),
                0,
            ),
            0,
        )

        entity_frequency[head] += total_count
        entity_frequency[tail] += total_count

        if head != tail:
            entity_neighbors[head].add(tail)
            entity_neighbors[tail].add(head)

    entity_connectivity = Counter(
        {
            entity: len(neighbors)
            for entity, neighbors
            in entity_neighbors.items()
        }
    )

    for entity in entity_frequency:
        entity_connectivity.setdefault(entity, 0)

    return entity_frequency, entity_connectivity


# graph data selection
def build_graph_data(
    relations: list[dict],
    entity_frequency: Counter,
    entity_connectivity: Counter,
    top_x: int,
    rank_by: str,
    require_both_entities: bool,
    selected_relation_types: list[str],
) -> dict:
    """
    Select top-ranked entities and relationships for the graph.

    Relationships themselves are not ranked.

    If require_both_entities is True:
        Both the head and tail must be in the Top X.

    If require_both_entities is False:
        At least one endpoint must be in the Top X.
    """

    if rank_by == "Connections":
        ranking = entity_connectivity
    else:
        ranking = entity_frequency

    top_entities_with_scores = ranking.most_common(top_x)

    top_entities = {
        entity
        for entity, _ in top_entities_with_scores
    }

    selected_relation_type_set = set(
        selected_relation_types
    )

    graph_edges = []

    for record in relations:
        head = str(record.get("head", "")).strip()
        tail = str(record.get("tail", "")).strip()
        relation_type = str(
            record.get("relation", "")
        ).strip()

        if not head or not tail or not relation_type:
            continue

        if relation_type not in selected_relation_type_set:
            continue

        if require_both_entities:
            should_include = (
                head in top_entities
                and tail in top_entities
            )
        else:
            should_include = (
                head in top_entities
                or tail in top_entities
            )

        if not should_include:
            continue

        graph_edges.append(
            {
                "head": head,
                "relation": relation_type,
                "tail": tail,
                "total_count": max(
                    safe_integer(
                        record.get("total_count"),
                        1,
                    ),
                    1,
                ),
                "transcript_count": max(
                    safe_integer(
                        record.get("transcript_count"),
                        0,
                    ),
                    0,
                ),
            }
        )

    graph_entities = set()

    for edge in graph_edges:
        graph_entities.add(edge["head"])
        graph_entities.add(edge["tail"])

    graph_nodes = []

    for entity in graph_entities:
        graph_nodes.append(
            {
                "entity": entity,
                "frequency": entity_frequency.get(
                    entity,
                    0,
                ),
                "connections": entity_connectivity.get(
                    entity,
                    0,
                ),
                "is_top_entity": entity in top_entities,
            }
        )

    return {
        "nodes": graph_nodes,
        "edges": graph_edges,
        "top_entities": top_entities,
        "top_entities_with_scores": (
            top_entities_with_scores
        ),
    }


# custom set of colors for relation types
def create_relation_colors(
    relation_types: list[str],
) -> dict[str, str]:
    """
    Assign one color to each relation type.
    """

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#393b79",
        "#637939",
        "#8c6d31",
        "#843c39",
        "#7b4173",
        "#3182bd",
        "#31a354",
        "#756bb1",
        "#636363",
        "#e6550d",
    ]

    return {
        relation_type: palette[
            index % len(palette)
        ]
        for index, relation_type
        in enumerate(sorted(relation_types))
    }


def create_pyvis_network(
    graph_data: dict,
    rank_by: str,
) -> tuple[Network, dict[str, str]]:
    """
    Convert selected nodes and edges into a PyVis network.

    Physics is used only for the initial layout. Custom zoom
    controls are added later and recreated whenever Streamlit
    rerenders the graph.
    """

    network = Network(
        height=f"{GRAPH_HEIGHT}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111111",
        cdn_resources="in_line",
    )

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    top_entities = graph_data["top_entities"]

    relation_types = sorted(
        {
            edge["relation"]
            for edge in edges
        }
    )

    relation_colors = create_relation_colors(
        relation_types
    )

    if rank_by == "Connections":
        ranking_field = "connections"
        ranking_label = "connections"
    else:
        ranking_field = "frequency"
        ranking_label = "appearances"

    maximum_node_score = max(
        (
            node[ranking_field]
            for node in nodes
        ),
        default=1,
    )

    maximum_node_score = max(
        maximum_node_score,
        1,
    )

    for node in nodes:
        entity = node["entity"]
        ranking_score = node[ranking_field]

        relative_size = math.sqrt(
            ranking_score / maximum_node_score
        )

        # Large minimum size so labels fit comfortably.
        node_size = 72 + (42 * relative_size)

        is_top_entity = entity in top_entities

        node_label = (
            f"{entity}\n"
            f"{ranking_score:,} {ranking_label}"
        )

        node_title = (
            f"<b>{html.escape(entity)}</b><br>"
            f"Entity appearances: "
            f"{node['frequency']:,}<br>"
            f"Unique connections: "
            f"{node['connections']:,}<br>"
            f"Top-ranked entity: "
            f"{'Yes' if is_top_entity else 'No'}"
        )

        network.add_node(
            entity,
            label=node_label,
            title=node_title,
            size=node_size,
            shape="circle",
            borderWidth=(
                4 if is_top_entity else 2
            ),
            color={
                "background": (
                    "#97c2fc"
                    if is_top_entity
                    else "#dedede"
                ),
                "border": (
                    "#2b7ce9"
                    if is_top_entity
                    else "#999999"
                ),
                "highlight": {
                    "background": "#ffd166",
                    "border": "#cc8b00",
                },
                "hover": {
                    "background": "#cce4ff",
                    "border": "#2b7ce9",
                },
            },
            font={
                "color": "#111111",
                "size": 14,
                "face": "Arial",
                "align": "center",
                "multi": False,
                "vadjust": 0,
            },
            margin={
                "top": 20,
                "right": 20,
                "bottom": 20,
                "left": 20,
            },
        )

    maximum_edge_count = max(
        (
            edge["total_count"]
            for edge in edges
        ),
        default=1,
    )

    maximum_edge_count = max(
        maximum_edge_count,
        1,
    )

    for edge in edges:
        edge_width = 2 + (
            7
            * math.log1p(
                edge["total_count"]
            )
            / math.log1p(
                maximum_edge_count
            )
        )

        edge_title = (
            f"<b>{html.escape(edge['relation'])}</b>"
            f"<br>"
            f"Head: {html.escape(edge['head'])}<br>"
            f"Tail: {html.escape(edge['tail'])}<br>"
            f"Total count: "
            f"{edge['total_count']:,}<br>"
            f"Transcript count: "
            f"{edge['transcript_count']:,}"
        )

        edge_color = relation_colors[
            edge["relation"]
        ]

        network.add_edge(
            edge["head"],
            edge["tail"],
            title=edge_title,
            color={
                "color": edge_color,
                "highlight": edge_color,
                "hover": edge_color,
            },
            width=edge_width,
            arrows="to",
        )

    network.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "navigationButtons": false,
            "keyboard": false,
            "multiselect": true,
            "tooltipDelay": 150,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          },
          "nodes": {
            "shape": "circle",
            "font": {
              "size": 14,
              "face": "Arial",
              "color": "#111111",
              "align": "center",
              "strokeWidth": 0
            },
            "margin": {
              "top": 20,
              "right": 20,
              "bottom": 20,
              "left": 20
            }
          },
          "edges": {
            "selectionWidth": 2,
            "hoverWidth": 1.5,
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.8
              }
            }
          },
          "physics": {
            "enabled": true,
            "solver": "barnesHut",
            "barnesHut": {
              "gravitationalConstant": -15000,
              "centralGravity": 0.12,
              "springLength": 280,
              "springConstant": 0.022,
              "damping": 0.5,
              "avoidOverlap": 1
            },
            "minVelocity": 1,
            "maxVelocity": 35,
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 50,
              "fit": true
            }
          }
        }
        """
    )

    return network, relation_colors


def add_graph_ui_to_html(
    graph_html: str,
    relation_colors: dict[str, str],
) -> str:
    """
    Add the legend and custom graph controls.

    The custom controls are bound directly to the current
    network object every time Streamlit rerenders the iframe.
    """

    legend_rows = []

    for relation, color in relation_colors.items():
        escaped_relation = html.escape(relation)

        legend_rows.append(
            f"""
            <div class="legend-row">
                <span
                    class="legend-line"
                    style="background: {color};"
                ></span>
                <span>{escaped_relation}</span>
            </div>
            """
        )

    legend_contents = "\n".join(legend_rows)

    additions = f"""
        <style>
            html,
            body {{
                margin: 0;
                padding: 0;
                background: #ffffff !important;
                color: #111111 !important;
                overflow: hidden;
            }}

            #mynetwork {{
                background: #ffffff !important;
                border: 1px solid #dedede !important;
                border-radius: 4px;
            }}

            .relation-legend {{
                position: fixed;
                right: 14px;
                top: 14px;
                z-index: 9999;
                min-width: 155px;
                max-width: 225px;
                max-height: 72%;
                overflow-y: auto;
                padding: 10px 12px;
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid #cfcfcf;
                border-radius: 8px;
                box-shadow:
                    0 2px 8px
                    rgba(0, 0, 0, 0.12);
                font-family: Arial, sans-serif;
                font-size: 12px;
                color: #111111;
            }}

            .legend-title {{
                margin-bottom: 8px;
                font-size: 14px;
                font-weight: 700;
            }}

            .legend-row {{
                display: flex;
                align-items: center;
                gap: 7px;
                margin: 6px 0;
            }}

            .legend-line {{
                display: inline-block;
                flex: 0 0 20px;
                width: 20px;
                height: 5px;
                border-radius: 2px;
            }}

            .graph-controls {{
                position: fixed;
                left: 14px;
                bottom: 14px;
                z-index: 10000;
                display: flex;
                flex-direction: column;
                gap: 5px;
            }}

            .graph-control-button {{
                width: 38px;
                height: 38px;
                padding: 0;
                background: rgba(255, 255, 255, 0.97);
                color: #111111;
                border: 1px solid #bdbdbd;
                border-radius: 5px;
                box-shadow:
                    0 1px 4px
                    rgba(0, 0, 0, 0.14);
                cursor: pointer;
                font-family: Arial, sans-serif;
                font-size: 22px;
                font-weight: 600;
                line-height: 1;
            }}

            .graph-control-button:hover {{
                background: #f0f0f0;
            }}

            .graph-control-button:active {{
                background: #e2e2e2;
            }}

            .graph-control-button.reset-button {{
                font-size: 17px;
            }}
        </style>

        <div class="relation-legend">
            <div class="legend-title">
                Relation Types
            </div>
            {legend_contents}
        </div>

        <div class="graph-controls">
            <button
                id="zoom-in-button"
                class="graph-control-button"
                type="button"
                aria-label="Zoom in"
                title="Zoom in"
            >
                +
            </button>

            <button
                id="zoom-out-button"
                class="graph-control-button"
                type="button"
                aria-label="Zoom out"
                title="Zoom out"
            >
                −
            </button>

            <button
                id="reset-view-button"
                class="graph-control-button reset-button"
                type="button"
                aria-label="Reset graph view"
                title="Reset graph view"
            >
                ⟳
            </button>
        </div>

        <script>
            document.addEventListener(
                "DOMContentLoaded",
                function () {{
                    const graphContainer =
                        document.getElementById(
                            "mynetwork"
                        );

                    const zoomInButton =
                        document.getElementById(
                            "zoom-in-button"
                        );

                    const zoomOutButton =
                        document.getElementById(
                            "zoom-out-button"
                        );

                    const resetViewButton =
                        document.getElementById(
                            "reset-view-button"
                        );

                    if (graphContainer) {{
                        graphContainer.addEventListener(
                            "wheel",
                            function (event) {{
                                event.preventDefault();
                                event.stopPropagation();
                            }},
                            {{
                                passive: false,
                                capture: true
                            }}
                        );
                    }}

                    function networkReady() {{
                        return (
                            typeof network !== "undefined"
                            && network !== null
                        );
                    }}

                    function getCurrentScale() {{
                        if (!networkReady()) {{
                            return 1;
                        }}

                        return network.getScale();
                    }}

                    function zoomToScale(newScale) {{
                        if (!networkReady()) {{
                            return;
                        }}

                        const boundedScale = Math.max(
                            0.05,
                            Math.min(newScale, 8)
                        );

                        network.moveTo({{
                            scale: boundedScale,
                            animation: {{
                                duration: 220,
                                easingFunction:
                                    "easeInOutQuad"
                            }}
                        }});
                    }}

                    if (zoomInButton) {{
                        zoomInButton.addEventListener(
                            "click",
                            function () {{
                                zoomToScale(
                                    getCurrentScale() * 1.25
                                );
                            }}
                        );
                    }}

                    if (zoomOutButton) {{
                        zoomOutButton.addEventListener(
                            "click",
                            function () {{
                                zoomToScale(
                                    getCurrentScale() / 1.25
                                );
                            }}
                        );
                    }}

                    if (resetViewButton) {{
                        resetViewButton.addEventListener(
                            "click",
                            function () {{
                                if (!networkReady()) {{
                                    return;
                                }}

                                network.fit({{
                                    animation: {{
                                        duration: 300,
                                        easingFunction:
                                            "easeInOutQuad"
                                    }}
                                }});
                            }}
                        );
                    }}

                    function finishInitialLayout() {{
                        if (!networkReady()) {{
                            return;
                        }}

                        network.setOptions({{
                            physics: {{
                                enabled: false
                            }}
                        }});

                        network.fit({{
                            animation: {{
                                duration: 300,
                                easingFunction:
                                    "easeInOutQuad"
                            }}
                        }});
                    }}

                    if (networkReady()) {{
                        network.once(
                            "stabilizationIterationsDone",
                            finishInitialLayout
                        );
                    }}

                    setTimeout(
                        finishInitialLayout,
                        1700
                    );
                }}
            );
        </script>
    """

    return graph_html.replace(
        "</body>",
        f"{additions}</body>",
    )


# load in local data
output_path = Path(OUTPUT_FILE)

if not output_path.exists():
    st.error(
        f"Could not find `{OUTPUT_FILE}`. "
        "Place the JSON file in the same directory "
        "as `app.py`."
    )
    st.stop()

try:
    relations = load_relations(
        str(output_path)
    )

except (
    OSError,
    json.JSONDecodeError,
    ValueError,
) as error:
    st.error(
        f"Could not load `{OUTPUT_FILE}`: "
        f"{error}"
    )
    st.stop()


entity_frequency, entity_connectivity = (
    calculate_entity_metrics(
        relations
    )
)

all_entities = (
    set(entity_frequency)
    | set(entity_connectivity)
)

all_relation_types = sorted(
    {
        str(
            record.get(
                "relation",
                "",
            )
        ).strip()
        for record in relations
        if str(
            record.get(
                "relation",
                "",
            )
        ).strip()
    }
)

# left sidebar
st.sidebar.title(
    "AIP Oral History Transcripts: "
    "Relation Knowledge Graph"
)

st.sidebar.header("Controls")

rank_by = st.sidebar.selectbox(
    "Rank entities by",
    options=[
        "Connections",
        "Frequency",
    ],
    index=0,
)

maximum_top_x = max(
    min(
        100,
        len(all_entities),
    ),
    5,
)

default_top_x = min(
    20,
    maximum_top_x,
)

top_x = st.sidebar.slider(
    "Top X entities",
    min_value=5,
    max_value=maximum_top_x,
    value=default_top_x,
    step=1,
)

require_both_entities = st.sidebar.checkbox(
    "Require both entities in Top X",
    value=True,
)

selected_relation_types = (
    st.sidebar.multiselect(
        "Relation Types",
        options=all_relation_types,
        default=all_relation_types,
    )
)

st.sidebar.caption(
    f"Data Source: {OUTPUT_FILE}"
)



graph_data = build_graph_data(
    relations=relations,
    entity_frequency=entity_frequency,
    entity_connectivity=entity_connectivity,
    top_x=top_x,
    rank_by=rank_by,
    require_both_entities=(
        require_both_entities
    ),
    selected_relation_types=(
        selected_relation_types
    ),
)

network, relation_colors = (
    create_pyvis_network(
        graph_data=graph_data,
        rank_by=rank_by,
    )
)


# summary values above visualization
summary_column_1, \
    summary_column_2, \
    summary_column_3, \
    summary_column_4 = st.columns(4)

summary_column_1.metric(
    label="Relations in JSON",
    value=f"{len(relations):,}",
)

summary_column_2.metric(
    label="Entities in corpus",
    value=f"{len(all_entities):,}",
)

summary_column_3.metric(
    label="Nodes displayed",
    value=f"{len(graph_data['nodes']):,}",
)

summary_column_4.metric(
    label="Edges displayed",
    value=f"{len(graph_data['edges']):,}",
)


# main graph visualization
st.caption("Zoom in to see entity names and scores.")

if not graph_data["edges"]:
    st.warning(
        "No relationships match the current controls. "
        "Try increasing Top X, selecting additional "
        "relation types, or turning off the requirement "
        "that both entities be in the Top X."
    )

else:
    graph_html = network.generate_html(
        name="knowledge_graph.html",
        notebook=False,
    )

    graph_html = add_graph_ui_to_html(
        graph_html=graph_html,
        relation_colors=relation_colors,
    )

    graph_component_key = (
        f"knowledge-graph-"
        f"{rank_by}-"
        f"{top_x}-"
        f"{require_both_entities}-"
        f"{hash(tuple(selected_relation_types))}"
    )

    components.html(
        graph_html,
        height=GRAPH_HEIGHT + 10,
        scrolling=False
    )


# top-ranked entities table
st.subheader(
    f"Top Entities Ranked by {rank_by}"
)

ranking_rows = []

for rank, (
    entity,
    ranking_score,
) in enumerate(
    graph_data[
        "top_entities_with_scores"
    ],
    start=1,
):
    ranking_rows.append(
        {
            "Rank": rank,
            "Entity": entity,
            "Entity appearances": (
                entity_frequency.get(
                    entity,
                    0,
                )
            ),
            "Unique connections": (
                entity_connectivity.get(
                    entity,
                    0,
                )
            ),
        }
    )

ranking_dataframe = pd.DataFrame(
    ranking_rows
)

st.dataframe(
    ranking_dataframe,
    hide_index=True,
    use_container_width=True,
    height=260,
)


# displayed relationships table
st.subheader(
    "Displayed Relationships"
)

relationship_dataframe = pd.DataFrame(
    graph_data["edges"]
)

if relationship_dataframe.empty:
    st.info(
        "No relationships are displayed "
        "under the current settings."
    )

else:
    relationship_dataframe = (
        relationship_dataframe.rename(
            columns={
                "head": "Head Entity",
                "relation": "Relation",
                "tail": "Tail Entity",
                "total_count": "Total Count",
                "transcript_count": (
                    "Transcript Count"
                ),
            }
        )
    )

    relationship_dataframe = (
        relationship_dataframe.sort_values(
            by=[
                "Total Count",
                "Transcript Count",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    st.dataframe(
        relationship_dataframe,
        hide_index=True,
        use_container_width=True,
        height=400,
    )
    