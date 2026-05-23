
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import Counter
from networkx.algorithms import community as nx_community

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# paths
SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_CLEAN   = SCRIPT_DIR.parent / "data_clean"
OUTPUT_DIR   = SCRIPT_DIR.parent / "outputs" / "nlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGES_PATH    = DATA_CLEAN / "network_edges_balanced.csv"
COMMENTS_PATH = DATA_CLEAN / "comments_master_processed_balanced.csv"

# visual style settings 
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.grid":        True,
    "grid.color":       "#eeeeee",
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
})

C_PROAI  = "#4C9BE8"   # blue  — Pro-AI
C_ANTIAI = "#E8724C"   # orange — Anti-AI
C_MIXED  = "#7C5CBF"   # purple — cross-group

VADER = SentimentIntensityAnalyzer()


#data loading and graph construction functions

def load_data():
    print("Loading data …")
    edges_df    = pd.read_csv(EDGES_PATH)
    comments_df = pd.read_csv(COMMENTS_PATH)

    
    replace_map = {"ProAI": "Pro-AI", "Anti AI": "Anti-AI"}
    for df in [edges_df, comments_df]:
        if "video_type" in df.columns:
            df["video_type"] = df["video_type"].replace(replace_map)

    if "vader_compound" not in comments_df.columns:
        print("  Computing VADER scores …")
        comments_df["vader_compound"] = comments_df["comment_text"].apply(
            lambda t: VADER.polarity_scores(str(t))["compound"]
        )

    print(f"  Edges: {len(edges_df):,}  |  Comments: {len(comments_df):,}")
    return edges_df, comments_df


def build_graphs(edges_df):
    # Fix: normalise video_type first
    edges_df = edges_df.copy()
    edges_df["video_type"] = edges_df["video_type"].replace({
        "AI": "Pro-AI",
        "Human": "Anti-AI",
        # add whatever values your CSV actually uses
    })
    
    def make(sub):
        return nx.from_pandas_edgelist(
            sub, "Source_User_ID", "Target_User_ID",
            edge_attr=["video_type"], create_using=nx.DiGraph()
        )
    G_full  = make(edges_df)
    G_proai = make(edges_df[edges_df["video_type"] == "Pro-AI"])
    G_anti  = make(edges_df[edges_df["video_type"] == "Anti-AI"])
    return G_full, G_proai, G_anti


def graph_stats(G):
    ug      = G.to_undirected()
    comps   = list(nx.connected_components(ug))
    degrees = [d for _, d in G.degree()]
    return {
        "Nodes":             G.number_of_nodes(),
        "Edges":             G.number_of_edges(),
        "Density (×10⁻³)":  round(nx.density(G) * 1000, 4),
        "Avg Degree":        round(np.mean(degrees), 2) if degrees else 0,
        "Max Degree":        max(degrees) if degrees else 0,
        "Components":        len(comps),
        "Largest Component": max(len(c) for c in comps) if comps else 0,
    }


def node_type_map(G_full, G_proai, G_anti):
    pro_nodes  = set(G_proai.nodes())
    anti_nodes = set(G_anti.nodes())

    ntype = {}
    for n in G_full.nodes():
        in_pro  = n in pro_nodes
        in_anti = n in anti_nodes
        if in_pro and in_anti:
            ntype[n] = "Mixed"
        elif in_pro:
            ntype[n] = "Pro-AI"
        else:
            ntype[n] = "Anti-AI"
    return ntype

def sentiment_map(comments_df):
    return (
        comments_df.groupby("author_channel_id")["vader_compound"]
        .mean().to_dict()
    )


#figure 1: global topology stats comparison

def fig1_global_topology(G_proai, G_anti):
    print("→ Fig 1: Global Topology …")

    s_pro  = graph_stats(G_proai)
    s_anti = graph_stats(G_anti)

    large_metrics = ["Nodes", "Edges", "Largest Component"]
    small_metrics = ["Density (×10⁻³)", "Avg Degree", "Max Degree", "Components"]

    fig = plt.figure(figsize=(14, 8))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1.2], hspace=0.6, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    for ax, metrics in [(ax1, large_metrics), (ax2, small_metrics)]:
        x     = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, [s_pro[m]  for m in metrics], width,
               label="Pro-AI",  color=C_PROAI,  edgecolor="white")
        ax.bar(x + width/2, [s_anti[m] for m in metrics], width,
               label="Anti-AI", color=C_ANTIAI, edgecolor="white")
        for i, m in enumerate(metrics):
            ax.text(i - width/2, s_pro[m]  * 1.05, f"{s_pro[m]:,}",
                    ha="center", va="bottom", fontsize=8)
            ax.text(i + width/2, s_anti[m] * 1.05, f"{s_anti[m]:,}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylabel("Value")
        ax.legend(framealpha=0.9, fontsize=8)
        ax.set_axisbelow(True)

    # network construction metadata panel
    ax3.axis("off")
    info_lines = [
        ("Nodes",  "YouTube users who posted or received at least one reply"),
        ("Edges",  "Directed reply interactions (User A replied to User B)"),
        ("Type",   "Directed (DiGraph): reply direction is meaningful for information flow analysis"),
        ("Weight", "Unweighted: edge presence captures interaction structure; frequency not required for topology"),
        ("Filter", "Thread level balanced sampling, isolated nodes retained, no degree threshold applied"),
        ("Data",   f"From: network_edges_balanced.csv,  {s_pro['Edges']+s_anti['Edges']:,} total edges,  {s_pro['Nodes']+s_anti['Nodes']:,} unique users across both subgraphs"),
    ]
    col_x = [0.01, 0.10]
    row_y = np.linspace(0.75, 0.05, len(info_lines))

    ax3.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax3.transAxes,
                                 facecolor="#f4f7fb", zorder=0))
    ax3.text(0.5, 0.93, "Network Construction Summary",
             ha="center", va="top", fontsize=9, fontweight="bold",
             transform=ax3.transAxes, color="#2E4057")
    for (label, desc), y in zip(info_lines, row_y):
        ax3.text(col_x[0], y, f"{label}:", fontsize=8, fontweight="bold",
                 transform=ax3.transAxes, color="#333", va="center")
        ax3.text(col_x[1], y, desc, fontsize=8,
                 transform=ax3.transAxes, color="#555", va="center")

    fig.suptitle("Fig 1: Global Network Topology: Pro-AI vs Anti-AI",
                 fontweight="bold", fontsize=13, y=0.99)
    out = OUTPUT_DIR / "fig1_global_topology.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   Saved {out.name}")
# degree distribution plots (log-log) for both subgraphs, with power-law fit lines

def fig2_degree_distribution(G_proai, G_anti):
    print("→ Fig 2: Degree Distribution …")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, G, label, color in [
        (axes[0], G_proai, "Pro-AI",  C_PROAI),
        (axes[1], G_anti,  "Anti-AI", C_ANTIAI),
    ]:
        degrees = sorted([d for _, d in G.degree()], reverse=True)
        counts  = Counter(degrees)
        xs      = sorted(counts.keys())
        ys      = [counts[x] for x in xs]

        ax.loglog(xs, ys, "o", color=color, markersize=4, alpha=0.7, label="Observed")

        # power-law fit line
        log_x = np.log(xs)
        log_y = np.log(ys)
        if len(log_x) > 2:
            coeff = np.polyfit(log_x, log_y, 1)
            fit_y = np.exp(np.poly1d(coeff)(log_x))
            ax.loglog(xs, fit_y, "--", color="#555", linewidth=1.5,
                      label=f"Power-law fit (γ={-coeff[0]:.2f})")

        ax.set_title(f"{label} Network — Degree Distribution", fontweight="bold")
        ax.set_xlabel("Degree (log scale)")
        ax.set_ylabel("Count (log scale)")
        ax.legend(fontsize=8)
        ax.set_axisbelow(True)

    fig.suptitle("Fig 2: Log-Log Degree Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_degree_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   Saved {out.name}")


# community detection and composition analysis: greedy modularity communities on the full graph, then composition of top communities by node type

def fig3_community_composition(G_full, ntype_map):
    print("Fig 3: Community Composition …")

    UG = G_full.to_undirected()
    communities = list(nx_community.greedy_modularity_communities(UG))
    communities.sort(key=len, reverse=True)
    top_n = min(8, len(communities))

    labels, pro_pct, anti_pct, mix_pct, sizes = [], [], [], [], []
    for i, comm in enumerate(communities[:top_n]):
        types = [ntype_map.get(n, "Anti-AI") for n in comm]
        total = len(types)
        labels.append(f"C{i+1}\n(n={total:,})")
        pro_pct.append(types.count("Pro-AI")  / total * 100)
        anti_pct.append(types.count("Anti-AI") / total * 100)
        mix_pct.append(types.count("Mixed")    / total * 100)
        sizes.append(total)

    x     = np.arange(top_n)
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x, pro_pct,  width, label="Pro-AI",  color=C_PROAI)
    b2 = ax.bar(x, anti_pct, width, bottom=pro_pct, label="Anti-AI", color=C_ANTIAI)
    b3 = ax.bar(x, mix_pct,  width,
                bottom=[p+a for p,a in zip(pro_pct, anti_pct)],
                label="Mixed", color=C_MIXED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Node Composition (%)")
    ax.set_title("Fig 3: Community Composition: Top Communities by Node Type",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 110)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_community_composition.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   Saved {out.name}")


# top hub nodes by in-degree, coloured by type (Pro-AI / Anti-AI / Mixed)

def fig4_top_hubs(G_full, ntype_map):
    print("→ Fig 4: Top Hub Nodes")
 
    in_deg  = dict(G_full.in_degree())
    out_deg = dict(G_full.out_degree())
    top20   = sorted(in_deg, key=in_deg.get, reverse=True)[:20]
    vals      = [in_deg[n]  for n in top20]
    out_vals  = [out_deg[n] for n in top20]
    colors    = [{"Pro-AI": C_PROAI, "Anti-AI": C_ANTIAI, "Mixed": C_MIXED}
                 .get(ntype_map.get(n, "Anti-AI"), C_ANTIAI) for n in top20]
    labels    = [f"User {i+1}" for i in range(20)]
 
    legend_patches = [
        mpatches.Patch(color=C_PROAI,  label="Pro-AI community"),
        mpatches.Patch(color=C_ANTIAI, label="Anti-AI community"),
        mpatches.Patch(color=C_MIXED,  label="Mixed / cross-community"),
    ]
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [2, 1]})
 
    # left: top 20 hubs by in-degree
    ax1.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    ax1.legend(handles=legend_patches, loc="lower right", framealpha=0.9, fontsize=8)
    ax1.set_xlabel("In-Degree (replies received)")
    ax1.set_title("(a) Top 20 Hubs by In-Degree", fontweight="bold")
    ax1.set_axisbelow(True)
 
    #right: in-degree vs out-degree — broadcaster vs engager roles ─────────
    scatter_colors = [{"Pro-AI": C_PROAI, "Anti-AI": C_ANTIAI, "Mixed": C_MIXED}
                       .get(ntype_map.get(n, "Anti-AI"), C_ANTIAI) for n in top20]
    ax2.scatter(out_vals, vals, c=scatter_colors, s=65, alpha=0.85,
                edgecolors="white", linewidths=0.5)
    for x, y, lbl in zip(out_vals, vals, labels):
        ax2.annotate(lbl, (x, y), fontsize=6, ha="left", va="bottom",
                     xytext=(3, 2), textcoords="offset points", color="#444")
 
    med_in  = np.median(vals)
    med_out = np.median(out_vals)
    ax2.axhline(med_in,  color="#aaa", linewidth=0.8, linestyle="--")
    ax2.axvline(med_out, color="#aaa", linewidth=0.8, linestyle="--")
 
    # quadrant labels
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    ax2.text(med_out * 0.1 if med_out > 0 else xlim[0] + 0.5,
             med_in  * 1.02,
             "Broadcaster\n(high in, low out)", fontsize=7, color="#666", va="bottom")
    ax2.text(med_out * 1.1 if med_out > 0 else xlim[0] + 0.5,
             med_in  * 0.15,
             "Engager\n(low in, high out)",    fontsize=7, color="#666", va="bottom")
 
    ax2.set_xlabel("Out-Degree (replies sent)")
    ax2.set_ylabel("In-Degree (replies received)")
    ax2.set_title("(b) User Role: Broadcaster vs Engager", fontweight="bold")
    ax2.legend(handles=legend_patches, fontsize=7, framealpha=0.9)
    ax2.set_axisbelow(True)
 
    fig.suptitle("Fig 4: Top 20 Hub Nodes: In-Degree & User Role Analysis",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_top_hubs.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   Saved {out.name}")


# sample of the network visualised with spring layout, nodes coloured by type and sized by degree
def fig5_network_sample(G_full, ntype_map):
    print("→ Fig 5: Network Sample Visualisation")

    # keep only the giant component, then sample 400 nodes
    UG    = G_full.to_undirected()
    giant = max(nx.connected_components(UG), key=len)
    sub_nodes = list(giant)
    rng   = np.random.default_rng(42)
    if len(sub_nodes) > 400:
        sub_nodes = list(rng.choice(sub_nodes, 400, replace=False))

    SG   = G_full.subgraph(sub_nodes)
    pos  = nx.spring_layout(SG, seed=42, k=0.4)

    node_colors = [
        {"Pro-AI": C_PROAI, "Anti-AI": C_ANTIAI, "Mixed": C_MIXED}
        .get(ntype_map.get(n, "Anti-AI"), C_ANTIAI)
        for n in SG.nodes()
    ]
    degrees     = dict(SG.degree())
    node_sizes  = [10 + degrees[n] * 8 for n in SG.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#f9f9f9")
    nx.draw_networkx_edges(SG, pos, ax=ax, alpha=0.12, edge_color="#999",
                           arrows=True, arrowsize=6, width=0.6)
    nx.draw_networkx_nodes(SG, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85)

    legend_patches = [
        mpatches.Patch(color=C_PROAI,  label="Pro-AI community"),
        mpatches.Patch(color=C_ANTIAI, label="Anti-AI community"),
        mpatches.Patch(color=C_MIXED,  label="Mixed / cross-community"),
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              framealpha=0.95, fontsize=10)
    ax.set_title("Fig 5: Reply Interaction Network\n"
                 "Node size ∝ degree; colour = community type",
                 fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig5_network_sample.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   Saved {out.name}")


# synthesis section: centrality vs sentiment scatter, hub sentiment heatmap, cross-group bridge users
def fig6_centrality_vs_sentiment(G_full, ntype_map, sent_map):
    print("→ Fig 6: Centrality vs Sentiment Scatter")

    in_deg  = dict(G_full.in_degree())
    nodes   = [n for n in G_full.nodes() if n in sent_map and in_deg[n] > 0]

    xs      = [in_deg[n]   for n in nodes]
    ys      = [sent_map[n] for n in nodes]
    colors  = [{"Pro-AI": C_PROAI, "Anti-AI": C_ANTIAI, "Mixed": C_MIXED}
               .get(ntype_map.get(n, "Anti-AI"), C_ANTIAI) for n in nodes]

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(xs, ys, c=colors, alpha=0.55, s=25, edgecolors="none")

    # trend lines per group
    for label, color in [("Pro-AI", C_PROAI), ("Anti-AI", C_ANTIAI)]:
        grp_nodes = [n for n in nodes if ntype_map.get(n) == label]
        gx = np.array([in_deg[n]   for n in grp_nodes])
        gy = np.array([sent_map[n] for n in grp_nodes])
        if len(gx) > 5:
            coeff = np.polyfit(gx, gy, 1)
            x_line = np.linspace(gx.min(), gx.max(), 100)
            ax.plot(x_line, np.poly1d(coeff)(x_line), "--",
                    color=color, linewidth=2, label=f"{label} trend")

    ax.axhline(0, color="#888", linewidth=0.8, linestyle=":")
    ax.axhline(0.05,  color="#bbb", linewidth=0.6, linestyle=":")
    ax.axhline(-0.05, color="#bbb", linewidth=0.6, linestyle=":")

    legend_patches = [
        mpatches.Patch(color=C_PROAI,  label="Pro-AI"),
        mpatches.Patch(color=C_ANTIAI, label="Anti-AI"),
        mpatches.Patch(color=C_MIXED,  label="Mixed"),
    ]
    handles, lbs = ax.get_legend_handles_labels()
    ax.legend(handles=legend_patches + handles, labels=["Pro-AI","Anti-AI","Mixed"] + lbs,
              fontsize=8, framealpha=0.9)

    ax.set_xlabel("In-Degree Centrality (Replies Received)")
    ax.set_ylabel("Mean VADER Compound Score")
    ax.set_title("Fig 6: Centrality vs Mean Sentiment by Community Type",
                 fontweight="bold", pad=12)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig6_centrality_vs_sentiment.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   Saved {out.name}")



# heatmap of sentiment for top hub nodes (top 30 by in-degree), sorted by sentiment, with bars coloured by community type
def fig7_hub_sentiment_heatmap(G_full, ntype_map, sent_map):
    print("→ Fig 7: Hub Sentiment Heatmap …")

    in_deg = dict(G_full.in_degree())
    top30  = sorted(in_deg, key=in_deg.get, reverse=True)[:30]
    top30  = [n for n in top30 if n in sent_map]

    labels   = [f"User {i+1}" for i in range(len(top30))]
    sentiments = [sent_map[n] for n in top30]
    node_type  = [ntype_map.get(n, "Anti-AI") for n in top30]

    # sort by sentiment
    order      = np.argsort(sentiments)
    sentiments = [sentiments[i] for i in order]
    node_type  = [node_type[i]  for i in order]
    labels     = [labels[i]     for i in order]

    bar_colors = [{"Pro-AI": C_PROAI, "Anti-AI": C_ANTIAI, "Mixed": C_MIXED}
                  .get(t, C_ANTIAI) for t in node_type]

    # heatmap row
    fig, (ax_bar, ax_heat) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw={"height_ratios": [4, 0.6]}, sharex=False
    )

    ax_bar.barh(range(len(labels)), sentiments, color=bar_colors, edgecolor="white")
    ax_bar.axvline(0,     color="#888", linewidth=0.9, linestyle=":")
    ax_bar.axvline(0.05,  color="#bbb", linewidth=0.7, linestyle="--")
    ax_bar.axvline(-0.05, color="#bbb", linewidth=0.7, linestyle="--")
    ax_bar.set_yticks(range(len(labels)))
    ax_bar.set_yticklabels(labels, fontsize=8)
    ax_bar.set_xlabel("Mean VADER Compound Score")
    ax_bar.set_title("Fig 7: Hub Node Sentiment (Top 30 by In Degree, Sorted by Sentiment)",
                     fontweight="bold", pad=12)
    ax_bar.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=C_PROAI,  label="Pro-AI"),
        mpatches.Patch(color=C_ANTIAI, label="Anti-AI"),
        mpatches.Patch(color=C_MIXED,  label="Mixed"),
    ]
    ax_bar.legend(handles=legend_patches, loc="lower right", framealpha=0.9, fontsize=8)

    # mini heatmap strip showing sentiment intensity
    heat_data = np.array(sentiments).reshape(1, -1)
    ax_heat.imshow(heat_data, aspect="auto",
                   cmap="RdYlGn", vmin=-1, vmax=1)
    ax_heat.set_yticks([])
    ax_heat.set_xticks([])
    ax_heat.set_xlabel("← More Negative          More Positive →", fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig7_hub_sentiment_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   Saved {out.name}")

#cross-group bridge users: identify top 20 "Mixed" nodes by in-degree, show their sentiment and number of interactions with each community (Pro-AI vs Anti-AI), visualised as a combined bar + scatter plot

def fig8_cross_group_bridges(G_full, ntype_map, sent_map, edges_df):
    print("→ Fig 8: Cross-Group Bridge Users")

    mixed_nodes = [n for n, t in ntype_map.items() if t == "Mixed" and n in sent_map]

    if len(mixed_nodes) == 0:
        print("   No mixed nodes found, skipping.")
        return

    in_deg = dict(G_full.in_degree())
    mixed_nodes.sort(key=lambda n: in_deg.get(n, 0), reverse=True)
    top_mixed = mixed_nodes[:20]

    labels     = [f"Bridge {i+1}" for i in range(len(top_mixed))]
    sentiments = [sent_map[n] for n in top_mixed]
    degrees    = [in_deg.get(n, 0) for n in top_mixed]

    # count interactions per community
    pro_int, anti_int = [], []
    for n in top_mixed:
        pro_e  = len(edges_df[
            ((edges_df["Source_User_ID"]==n) | (edges_df["Target_User_ID"]==n))
            & (edges_df["video_type"]=="Pro-AI")
        ])
        anti_e = len(edges_df[
            ((edges_df["Source_User_ID"]==n) | (edges_df["Target_User_ID"]==n))
            & (edges_df["video_type"]=="Anti-AI")
        ])
        pro_int.append(pro_e)
        anti_int.append(anti_e)

    x     = np.arange(len(top_mixed))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # top panel: interactions per community
    ax1.bar(x - width/2, pro_int,  width, label="Pro-AI interactions",  color=C_PROAI)
    ax1.bar(x + width/2, anti_int, width, label="Anti-AI interactions", color=C_ANTIAI)
    ax1.set_ylabel("No. of Interactions")
    ax1.set_title("Fig 8: Cross-Group Bridge Users: Interactions & Sentiment",
                  fontweight="bold", pad=12)
    ax1.legend(framealpha=0.9)
    ax1.set_axisbelow(True)

    # bottom panel: sentiment of each bridge user
    bar_colors = ["#4CAF50" if s >= 0.05 else "#F44336" if s <= -0.05 else "#FFC107"
                  for s in sentiments]
    ax2.bar(x, sentiments, color=bar_colors, edgecolor="white")
    ax2.axhline(0,     color="#888", linewidth=0.9, linestyle=":")
    ax2.axhline(0.05,  color="#bbb", linewidth=0.7, linestyle="--")
    ax2.axhline(-0.05, color="#bbb", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("Mean VADER Compound")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_axisbelow(True)

    sent_patches = [
        mpatches.Patch(color="#4CAF50", label="Positive (≥0.05)"),
        mpatches.Patch(color="#FFC107", label="Neutral"),
        mpatches.Patch(color="#F44336", label="Negative (≤-0.05)"),
    ]
    ax2.legend(handles=sent_patches, loc="lower right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig8_cross_group_bridges.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"   Saved {out.name}")


# main function to run all steps

def main():
    edges_df, comments_df = load_data()
    print(edges_df["video_type"].value_counts())
    G_full, G_proai, G_anti = build_graphs(edges_df)
    ntype_map = node_type_map(G_full, G_proai, G_anti)
    sent_map  = sentiment_map(comments_df)

    print(f"\nGraph ready — {G_full.number_of_nodes():,} nodes, "
          f"{G_full.number_of_edges():,} edges\n")

    # graph models and basic stats
    fig1_global_topology(G_proai, G_anti)
    fig2_degree_distribution(G_proai, G_anti)
    fig3_community_composition(G_full, ntype_map)
    fig4_top_hubs(G_full, ntype_map)
    fig5_network_sample(G_full, ntype_map)

    # synthesis section: centrality vs sentiment scatter
    fig6_centrality_vs_sentiment(G_full, ntype_map, sent_map)
    fig7_hub_sentiment_heatmap(G_full, ntype_map, sent_map)
    fig8_cross_group_bridges(G_full, ntype_map, sent_map, edges_df)

    print(f"\n✓ All figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()