"""
Generates a professional multi-page .drawio file for the Horse Segmentation project.
Can be opened in:
  - VS Code (via 'Draw.io Integration' extension by Henning Dieterichs)
  - app.diagrams.net (official web interface)
  - Draw.io Desktop application
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def create_drawio_file(filename="horse_segmentation.drawio"):
    mxfile = ET.Element("mxfile", host="app.diagrams.net", version="21.0.0", type="device")
    
    # -------------------------------------------------------------
    # DIAGRAM 1: MODEL ARCHITECTURES
    # -------------------------------------------------------------
    diag1 = ET.SubElement(mxfile, "diagram", id="architectures", name="1. Model Architectures")
    model1 = ET.SubElement(diag1, "mxGraphModel", dx="1600", dy="1100", grid="1", gridSize="10",
                           guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1",
                           pageScale="1", pageWidth="1750", pageHeight="1150", math="0", shadow="0")
    root1 = ET.SubElement(model1, "root")
    ET.SubElement(root1, "mxCell", id="0")
    ET.SubElement(root1, "mxCell", id="1", parent="0")
    
    cell_id = 2
    def get_id():
        nonlocal cell_id
        cid = str(cell_id)
        cell_id += 1
        return cid

    def add_cell(parent, value, style, x, y, w, h):
        cid = get_id()
        c = ET.SubElement(parent, "mxCell", id=cid, value=value, style=style, vertex="1", parent="1")
        geo = ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h))
        geo.set("as", "geometry")
        return cid

    def add_edge(parent, value, style, source, target, points=None):
        cid = get_id()
        e = ET.SubElement(parent, "mxCell", id=cid, value=value, style=style, edge="1", parent="1", source=source, target=target)
        geo = ET.SubElement(e, "mxGeometry", relative="1")
        geo.set("as", "geometry")
        if points:
            pts = ET.SubElement(geo, "Array")
            pts.set("as", "points")
            for px, py in points:
                ET.SubElement(pts, "mxPoint", x=str(px), y=str(py))
        return cid

    # Main Title Header
    add_cell(root1, "<b>Horse Segmentation (Weizmann Dataset) — Neural Architecture Comparison</b><br><font style='font-size: 13px; font-weight: normal; color: #5f6368;'>Investigating the impact of Skip Connections on Boundary Delineation (Problem Statement 3)</font>",
             "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=20;fontColor=#202124;",
             40, 20, 1660, 50)

    # Architectural specs
    models_spec = [
        {
            "name": "Baseline (No Skip)",
            "subtitle": "Standard Hourglass Encoder-Decoder",
            "skip": "none",
            "up": "Bilinear + 1×1",
            "color_header": "#EA4335",
            "color_bg": "#FCE8E6",
            "x": 50,
            "params": "1.80M params | 87.81% IoU | 76.18% Boundary F1",
            "verdict": "⚠️ High boundary blur: Fine details (legs/ears) squeezed through 16×16 bottleneck."
        },
        {
            "name": "Proposed 1: Concat (U-Net)",
            "subtitle": "Direct Lateral Concatenation (U-Net style)",
            "skip": "concat",
            "up": "Bilinear + 1×1",
            "color_header": "#1A73E8",
            "color_bg": "#E8F0FE",
            "x": 470,
            "params": "2.10M params | 89.97% IoU | 83.51% Boundary F1",
            "verdict": "⭐ Best Performance: +7.33% Boundary F1 jump (p=3.4e-14), sharp leg/tail edges."
        },
        {
            "name": "Proposed 2: Additive Skip",
            "subtitle": "Element-Wise Addition (ResNet/FPN style)",
            "skip": "add",
            "up": "Bilinear + 1×1",
            "color_header": "#34A853",
            "color_bg": "#E6F4EA",
            "x": 890,
            "params": "1.80M params | 88.77% IoU | 82.69% Boundary F1",
            "verdict": "✅ Zero Parameter Overhead: Matches Baseline size while recovering +6.51% Boundary F1."
        },
        {
            "name": "Proposed 3: Transposed Conv",
            "subtitle": "Concat Skip + Learnable Deconvolution",
            "skip": "concat",
            "up": "TransConv 2×2",
            "color_header": "#FBBC04",
            "color_bg": "#FEF7E0",
            "x": 1310,
            "params": "2.10M params | 89.61% IoU | 82.48% Boundary F1",
            "verdict": "✅ Learned Upsampling: Similar accuracy to P1; minor checkerboard sensitivity."
        }
    ]

    chs = [32, 64, 128, 256]
    res = [256, 128, 64, 32]
    bottleneck_ch, bottleneck_res = 256, 16

    for spec in models_spec:
        bx = spec["x"]
        
        # Panel Background Container
        add_cell(root1, "",
                 f"rounded=1;whiteSpace=wrap;html=1;strokeColor={spec['color_header']};fillColor=#FFFFFF;strokeWidth=2;shadow=1;",
                 bx, 90, 390, 960)

        # Panel Header
        add_cell(root1, f"<b><font style='font-size: 15px;'>{spec['name']}</font></b><br><font style='font-size: 11px;'>{spec['subtitle']}</font>",
                 f"rounded=1;whiteSpace=wrap;html=1;fillColor={spec['color_bg']};strokeColor={spec['color_header']};strokeWidth=1;fontColor=#202124;",
                 bx + 15, 105, 360, 45)

        # Metrics Chip
        add_cell(root1, f"<b>{spec['params']}</b>",
                 "rounded=1;whiteSpace=wrap;html=1;fillColor=#F1F3F4;strokeColor=#BDC1C6;fontSize=10;fontColor=#3C4043;",
                 bx + 15, 158, 360, 26)

        # Input Node
        in_id = add_cell(root1, "<b>Input Image</b><br>3 × 256 × 256",
                         "rounded=1;whiteSpace=wrap;html=1;fillColor=#F8F9FA;strokeColor=#5F6368;fontSize=11;",
                         bx + 35, 195, 130, 42)

        # Output Node
        out_id = add_cell(root1, "<b>Binary Mask</b><br>1 × 256 × 256 (Sigmoid)",
                          "rounded=1;whiteSpace=wrap;html=1;fillColor=#E6F4EA;strokeColor=#137333;fontSize=11;fontColor=#137333;fontStyle=1;",
                          bx + 225, 195, 135, 42)

        prev_enc = in_id
        enc_ids = []
        dec_ids = []

        # Encoder Levels
        for i in range(4):
            ey = 255 + i * 110
            enc_id = add_cell(root1, f"<b>Enc Stage {i+1}</b><br>2× [3×3 Conv, BN, ReLU]<br><b>{chs[i]} ch @ {res[i]}²</b>",
                              "rounded=1;whiteSpace=wrap;html=1;fillColor=#D0E1FD;strokeColor=#1A73E8;fontSize=10;fontColor=#0D47A1;",
                              bx + 30, ey, 140, 52)
            enc_ids.append(enc_id)
            
            # Downsample arrow
            down_label = "↓ MaxPool 2×2" if i > 0 else "↓ Conv Block"
            add_edge(root1, down_label, "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#1A73E8;strokeWidth=1.5;fontSize=9;fontColor=#1A73E8;",
                     prev_enc, enc_id)
            prev_enc = enc_id

        # Bottleneck Node
        bot_id = add_cell(root1, f"<b>Bottleneck</b><br>2× [3×3 Conv, BN, ReLU]<br><b>{bottleneck_ch} ch @ {bottleneck_res}²</b>",
                          "rounded=1;whiteSpace=wrap;html=1;fillColor=#E1D5E7;strokeColor=#9673A6;fontSize=10;fontColor=#4A148C;fontStyle=1;",
                          bx + 115, 715, 160, 56)
        add_edge(root1, "↓ MaxPool 2×2", "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#1A73E8;strokeWidth=1.5;fontSize=9;fontColor=#1A73E8;",
                 prev_enc, bot_id)

        # Decoder Levels (Bottom-up)
        prev_dec = bot_id
        for i in reversed(range(4)):
            dy = 255 + i * 110
            dec_id = add_cell(root1, f"<b>Dec Stage {i+1}</b><br>2× [3×3 Conv, BN, ReLU]<br><b>{chs[i]} ch @ {res[i]}²</b>",
                              "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;fontSize=10;fontColor=#B45F06;",
                              bx + 220, dy, 140, 52)
            dec_ids.append(dec_id)

            # Upsample arrow
            up_txt = f"↑ {spec['up']}"
            add_edge(root1, up_txt, "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#D79B00;strokeWidth=1.5;fontSize=9;fontColor=#D79B00;",
                     prev_dec, dec_id)
            prev_dec = dec_id

            # Skip Connection (Horizontal)
            if spec["skip"] == "concat":
                add_edge(root1, "concat [‖]", "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#34A853;strokeWidth=2;dashed=1;fontSize=9;fontColor=#137333;fontStyle=1;",
                         enc_ids[i], dec_id)
            elif spec["skip"] == "add":
                add_edge(root1, "add [⊕]", "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#188038;strokeWidth=2;dashed=1;fontSize=9;fontColor=#137333;fontStyle=1;",
                         enc_ids[i], dec_id)
            elif spec["skip"] == "none":
                # Show an X mark or 'No Skip' indication
                pass

        # Final 1x1 conv to mask
        add_edge(root1, "1×1 Conv", "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#137333;strokeWidth=1.5;fontSize=9;fontColor=#137333;",
                 prev_dec, out_id)

        # Bottom verdict description box
        add_cell(root1, f"<b>Analysis:</b><br>{spec['verdict']}",
                 "rounded=1;whiteSpace=wrap;html=1;fillColor=#F8F9FA;strokeColor=#BDC1C6;fontSize=10;align=left;spacingLeft=8;fontColor=#3C4043;",
                 bx + 15, 800, 360, 65)

        # Key architectural takeaway
        if spec["skip"] == "none":
            takeaway = "<b>Bottleneck Loss:</b> Spatial dimensions reduced 16×. Thin horse features (legs, tail) are non-recoverable."
        elif spec["skip"] == "concat":
            takeaway = "<b>U-Net Feature Reuse:</b> Fine spatial gradients bypass bottleneck, allowing pixel-exact edge reconstruction."
        elif spec["skip"] == "add":
            takeaway = "<b>Efficient Residual Path:</b> Adds encoder feature map directly; no memory expansion from channel concat."
        else:
            takeaway = "<b>Learnable Reconstruction:</b> Deconvolution weights learn task-specific interpolation filters."
            
        add_cell(root1, takeaway,
                 f"rounded=1;whiteSpace=wrap;html=1;fillColor={spec['color_bg']};strokeColor={spec['color_header']};fontSize=10;align=left;spacingLeft=8;fontColor=#202124;",
                 bx + 15, 880, 360, 60)

    # Global Legend Box
    add_cell(root1, "<b>Architecture Legend:</b> &nbsp;&nbsp; "
                    "<font color='#1A73E8'>■ Encoder ConvBlock (32→64→128→256)</font> &nbsp;&nbsp;|&nbsp;&nbsp; "
                    "<font color='#9673A6'>■ Bottleneck (256ch @ 16×16)</font> &nbsp;&nbsp;|&nbsp;&nbsp; "
                    "<font color='#D79B00'>■ Decoder ConvBlock</font> &nbsp;&nbsp;|&nbsp;&nbsp; "
                    "<font color='#34A853'><b>- - ▶ Skip Connection (Concat / Add)</b></font>",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#F1F3F4;strokeColor=#BDC1C6;fontSize=12;fontColor=#202124;align=center;",
             50, 1070, 1650, 40)

    # -------------------------------------------------------------
    # DIAGRAM 2: END-TO-END ML PIPELINE
    # -------------------------------------------------------------
    diag2 = ET.SubElement(mxfile, "diagram", id="pipeline", name="2. End-to-End Pipeline")
    model2 = ET.SubElement(diag2, "mxGraphModel", dx="1600", dy="1100", grid="1", gridSize="10",
                           guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1",
                           pageScale="1", pageWidth="1700", pageHeight="1100", math="0", shadow="0")
    root2 = ET.SubElement(model2, "root")
    ET.SubElement(root2, "mxCell", id="0")
    ET.SubElement(root2, "mxCell", id="1", parent="0")

    # Title
    add_cell(root2, "<b>Horse Semantic Segmentation — End-to-End Experimental Pipeline</b><br><font style='font-size: 13px; font-weight: normal; color: #5f6368;'>From Raw Weizmann Dataset to Hyperparameter Tuning, Multi-Architecture Training, and Boundary Significance Testing</font>",
             "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=20;fontColor=#202124;",
             40, 20, 1620, 50)

    # Pipeline Stages Columns (5 Stages)
    stages = [
        {
            "num": "Stage 1",
            "title": "Data Ingestion & Split",
            "module": "src/data.py & src/eda.py",
            "color": "#1A73E8",
            "bg": "#E8F0FE",
            "x": 60,
            "steps": [
                ("Weizmann Horse DB", "327 Raw RGB images + 327 Ground Truth Binary Masks", "#FFFFFF"),
                ("Pairing & Resizing", "Match filenames, bilinear resize to 256×256, save as cache_256.npz (~64MB in RAM)", "#FFFFFF"),
                ("Stratified Split (70/15/15)", "Stratified by horse pixel ratio (bins=3):\n• Train: 228 images\n• Val: 49 images\n• Test: 50 images", "#FFFFFF"),
                ("Dataset Audit & EDA", "Distribution of sizes, aspect ratios, foreground share, and mean horse mask", "#FFFFFF")
            ]
        },
        {
            "num": "Stage 2",
            "title": "Hyperparameter Tuning",
            "module": "src/tuning.py",
            "color": "#9334E6",
            "bg": "#F3E8FD",
            "x": 380,
            "steps": [
                ("Validation-Only Guard", "Strict zero-leakage policy: Test split is locked and never touched during tuning", "#FFFFFF"),
                ("Phase A: Loss Selection", "Compare BCE vs Dice vs BCE+Dice across Baseline & P1 for 10-20 epochs.\nSelected: Compound BCE+Dice", "#FFFFFF"),
                ("Phase B: Learning Rate", "Grid sweep over candidate LRs (1e-3, 3e-4).\nSelected optimal shared LR: 3e-4", "#FFFFFF"),
                ("Tuning Registry", "Results logged to tune_summary.csv and used for final training runs", "#FFFFFF")
            ]
        },
        {
            "num": "Stage 3",
            "title": "Model Training",
            "module": "src/engine.py & src/models.py",
            "color": "#E37400",
            "bg": "#FEF7E0",
            "x": 700,
            "steps": [
                ("On-GPU Augmentation", "Vectorized batch transforms:\n• Horizontal Flip (p=0.5)\n• Affine Scale (0.85-1.15)\n• Rotation (±10°), Color Jitter", "#FFFFFF"),
                ("Mixed Precision (AMP)", "PyTorch torch.cuda.amp + GradScaler for fast 16-bit tensor operations on T4 GPU", "#FFFFFF"),
                ("Optimization Setup", "AdamW optimizer, weight decay 1e-4, batch size 8, 40 epochs (quick) / 120 (full)", "#FFFFFF"),
                ("Checkpointing", "Per-epoch evaluation on Val split. Saves best.pt based strictly on peak Val Dice score", "#FFFFFF")
            ]
        },
        {
            "num": "Stage 4",
            "title": "Test Evaluation & Ablation",
            "module": "src/evaluate.py & src/metrics.py",
            "color": "#D93025",
            "bg": "#FCE8E6",
            "x": 1020,
            "steps": [
                ("Single-Pass Test Eval", "Untouched 50 test images evaluated once with frozen best.pt checkpoints", "#FFFFFF"),
                ("Boundary Metrics Engine", "Morphological erosion extracts 1-pixel contour; computes Boundary F1 (tol=2px) & HD95", "#FFFFFF"),
                ("Skip Knockout Ablation", "Selectively zero out skip connections at stage 0, 1, 2, 3 to quantify individual impact", "#FFFFFF"),
                ("Efficiency Benchmark", "Measures latency (ms/image), parameters (M), FLOPs, and model size (MB)", "#FFFFFF")
            ]
        },
        {
            "num": "Stage 5",
            "title": "Statistical Analysis & Viz",
            "module": "src/analysis.py",
            "color": "#188038",
            "bg": "#E6F4EA",
            "x": 1340,
            "steps": [
                ("Hypothesis Testing", "Paired Student's t-test + Wilcoxon signed-rank test (p=3.4e-14, 96% images improved)", "#FFFFFF"),
                ("Bootstrap Confidence", "2,000 resamples to construct non-parametric 95% Confidence Intervals", "#FFFFFF"),
                ("10 Report Figures", "Generates training curves, error maps, boundary examples, qualitative grids, efficiency", "#FFFFFF"),
                ("Executive Summary", "Auto-compiles summary.md report and formatted CSV tables for submission", "#FFFFFF")
            ]
        }
    ]

    stage_cols = []
    for s in stages:
        sx = s["x"]
        # Column boundary
        add_cell(root2, "",
                 f"rounded=1;whiteSpace=wrap;html=1;strokeColor={s['color']};fillColor=#FFFFFF;strokeWidth=2;shadow=1;",
                 sx, 90, 290, 930)

        # Column Header
        add_cell(root2, f"<b><font style='font-size: 11px; color: {s['color']};'>{s['num'].upper()}</font></b><br><b><font style='font-size: 14px;'>{s['title']}</font></b><br><font style='font-size: 10px; color: #5f6368;'>{s['module']}</font>",
                 f"rounded=1;whiteSpace=wrap;html=1;fillColor={s['bg']};strokeColor={s['color']};strokeWidth=1;fontColor=#202124;",
                 sx + 15, 105, 260, 58)

        # Step cards
        prev_card = None
        card_ids = []
        for idx, (head, desc, bgc) in enumerate(s["steps"]):
            cy = 190 + idx * 195
            cid = add_cell(root2, f"<b><font style='font-size: 12px; color: {s['color']};'>{idx+1}. {head}</font></b><br><br><font style='font-size: 11px; color: #3C4043;'>{desc.replace(chr(10), '<br>')}</font>",
                           f"rounded=1;whiteSpace=wrap;html=1;fillColor={bgc};strokeColor=#DADCE0;strokeWidth=1;align=left;spacingLeft=10;spacingTop=6;spacingRight=10;",
                           sx + 15, cy, 260, 160)
            card_ids.append(cid)
            if prev_card:
                add_edge(root2, "", "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#BDC1C6;strokeWidth=1.5;",
                         prev_card, cid)
            prev_card = cid
        stage_cols.append(card_ids)

    # Inter-stage connector arrows
    for i in range(len(stages) - 1):
        add_edge(root2, "Passes Data / Config",
                 "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#202124;strokeWidth=2;fontSize=10;fontColor=#202124;fontStyle=1;",
                 stage_cols[i][len(stages[i]["steps"]) - 1], stage_cols[i+1][0],
                 points=[(stages[i]["x"] + 280, 880), (stages[i+1]["x"] + 10, 220)])

    # Bottom Takeaway Banner
    add_cell(root2, "<b>Key Empirical Conclusion:</b> &nbsp; While Region IoU improves moderately from <b>87.81%</b> to <b>89.97% (+2.16%)</b>, Boundary F1 achieves a dramatic boost from <b>76.18%</b> to <b>83.51% (+7.33%, p=3.4e-14)</b>. "
                    "This validates that the primary functional benefit of skip connections is spatial boundary reconstruction rather than interior area classification.",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F0FE;strokeColor=#1A73E8;strokeWidth=1.5;fontSize=12;fontColor=#174EA6;align=center;",
             60, 1040, 1570, 48)

    # Pretty-print XML and save
    xml_str = ET.tostring(mxfile, encoding="utf-8")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    print(f"✔ Successfully created draw.io file: {filename} ({os.path.getsize(filename):,} bytes)")

if __name__ == "__main__":
    create_drawio_file()

