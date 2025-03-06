import gradio as gr
from click import style
from spotSeg import spotSeg, spotLoadexample, ImageInfo
from cellseg import cellSeg, cellLoadExample
from cell_spotSeg import cellSpotSeg, cellSpotLoadExample
from pathlib import Path
import os
import subprocess
import time

css = """

.gradio-container .prose {  /* 定位Markdown渲染容器 */
    font-size: 18px !important;
    line-height: 1.7 !important;
}

.gradio-container .prose h1 {
    font-size: 1.8em !important;
    border-bottom: 2px solid #eee !important;
}

.gradio-container .prose h2 {
    font-size: 1.5em !important;
    color: #333 !important;
}

.gradio-container .prose h3 {
    font-size: 1.2em !important;
    color: #333 !important;
}

.gradio-container .prose ul {
    padding-left: 2em !important;
}


.gradio-container {
    max-width: none !important;
    margin: 0 !important;
    padding: 30px 200px 100px 200px !important;
    width: 100vw !important;
    min-height: 100vh !important;
    box-sizing: border-box !important;
    background: linear-gradient(
        135deg,
        #ffffff 0%,          /* 基底白 */
        #f8f7ff 15%,         /* 淡紫调 (RGB 248,247,255) */
        #ffffff 35%,         /* 核心白 */
        #f0f8ff 65%,         /* 空气蓝 (RGB 240,248,255) */
        #ffffff 85%,         /* 过渡白 */
        #f5f7ff 100%         /* 冰川蓝 (RGB 245,247,255) */
    ) !important;
    background-attachment: fixed;
}




#prompt_textbox div {
    background-color: #f7f7f7;
    color: #ff0000; !important;
}

.display-textbox {
    width: auto !important;
    max-width: 100%; /* 防止宽度超出屏幕 */
    display: inline-block;
    white-space: nowrap; /* 防止文本换行 */
}

#custom_button {
    color: blue;
}

button[role="tab"] {
    font-size: 20px;
    color: #333;
    
    border: none;
    padding: 10px 20px;
    margin: 15px;
    border-radius: 8px;
    transition: all 0.3s ease;
}

/* 选中 Table 的样式 */
button[aria-selected="true"] {
    font-size: 22px; /* 字体变大 */
    color: #fff !important; /* 字体颜色 */
    background: linear-gradient(45deg, #6a11cb, #2575fc); /* 背景渐变 */
    border-bottom: 3px solid #ffeb3b; /* 下划线 */
    transform: scale(1.05); /* 轻微放大 */
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* 阴影效果 */
}

/* 悬停效果 */
button[role="tab"]:hover {
    background-color: #ddd;
    color: #000;
}

.spot-video-container {
    margin: 50px 50px 50px 100px !important;  /* 上 20px，右 50px，下 30px，左 50px */
}

.cell-video-container {
    margin: 50px 100px 50px 50px !important;  /* 上 20px，右 50px，下 30px，左 50px */
}

.terminal {
    font-family: 'DejaVu Sans Mono', 'Courier New', monospace !important;
    font-size: 14px;
    background: #1E1E1E;
    color: #D4D4D4;
    padding: 12px;
    white-space: pre;       /* 保留空格和换行 */
    letter-spacing: 0;      /* 消除字符间距 */
    line-height: 1.2;
    tab-size: 8;            /* 制表符对齐 */
    overflow-x: auto;       /* 允许横向滚动 */
}


"""

markdown_content = """

## 1. Spot Segmentation

### Input Parameters
**1.Image**  
Fluorescence micrograph of target molecules (e.g., RNA, proteins). Accepts either:  
- Single-channel images captured at specific excitation wavelengths  
- Multi-channel merged images through wavelength fusion  

**2.Weight**  
Model weights trained on six distinct datasets:  
- First five datasets from:  *Eichenberger, B. T., Zhan, Y., Rempfler, M., Giorgetti, L. & Chao, J. A. deepBlink: threshold-independent detection and localization of diffraction-limited spots. Nucleic Acids Res. 49, 7292–7302 (2021)*  
- Sixth dataset: Proprietary experimental data from our research group  

**3.Scale Factor**  
- Preprocessing parameter for grid partition (Divide an image into blocks of factor * factor, make predictions, and then concatenate the results)  
- Recommended default: 4  
- Optimal range for 1024×1024 images: 4/8  

### Output Parameters
**1.Spot Number**  
Total count of fluorescent spot  

**2.Bbox**  
Instance segmentation boundaries superimposed on original image  

**3.Mask**  
Predicted spot contours visualization  

**4.Fluorescence Intensity Distribution**  
Histogram showing:  
- X-axis: Fluorescence intensity (a.u.)  
- Y-axis: Frequency distribution  

**5.Spatial Coordinate**  
Tabular data containing:  
- Centroid coordinates (x,y)  
- Pixel area measurements  

---

## 2. Cell Segmentation

### Input Parameters
**1.Image**  
Accepts:  
- Merged composite images (bright-field + DAPI + cytoplasmic fluorescence)  
- Whole-cell fluorescence staining micrographs  

**2.Weight**  
Single optimized weight file trained on proprietary cellular imaging dataset  

### Output Parameters
**1.Cell Number**  
Total count of cell of image

**2.Bbox**  
Cellular boundary mapping  

**3.mask**  
Morphological contour visualization  

---

## 3. Single-Cell Spot Quantification

### Input Parameters
**Requirements**  
- Maintains previous modules' parameter specifications  
- **Critical**: Spot and cellular images must share identical FOV  

### Output Parameters
**1.Spot number**  
Total spot count  
**2.Cell number**  
Total cell count  

**3.Spot Bbox**  
Instance segmentation boundaries superimposed on original image  

**4.Spot Mask**  
Predicted spot contours visualization  

**5.Cell Bbox**  
Cellular boundary mapping  

**6.Cell Mask**  
Morphological contour visualization  

**7.Tabel**  
Count the number of spots within each cell
"""

def imageShow(image):
    return image


def list_files():
    safe_dir = "./download"  # 指定安全目录
    return [f for f in os.listdir(safe_dir) if os.path.isfile(os.path.join(safe_dir, f))]

def run_nvidia_smi():
    while True:
        result = subprocess.run(
            ["nvitop"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
        output = result.stdout if result.stdout else result.stderr

        yield f'<div class="terminal">{output}</div>'
        time.sleep(1)


with gr.Blocks(title="Fluorescence image process system",  css=css) as demo:
    # gr.Markdown("<div style='text-align:center;'><h1>Fluorescence image process system<h1></div>")

    gr.Markdown("""
    <div style='
        text-align: left;
        color: #2980b9;
        font-family: "Algerian", "Z003", "P052", "Palatino Linotype", "Book Antiqua", serif;
        font-size: 3.0em;
        font-weight: 600;
        padding: 15px 0;
        margin-bottom: 25px;
        background: linear-gradient(45deg, #f0f8ff, #ffffff);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    '>
    🎋🎐🧸️ Fluorescence Image Processing
    </div>
    """)



    # gr.Markdown("## Spot segmentation subsystem")

    with gr.Tab("Home"):
        gr.HTML("""
                    <div style='
                        text-align: left;
                        font-family: system-ui, sans-serif;
                        font-size: 2.0em;
                        padding: 20px, 20px;
                        margin: 30px 0 25px 20px;
                        border: none !important;
                    '>
                    Welcome to FIP System
                    </div>
                """)

        with gr.Row():
            with gr.Column(scale=10):
                spotVideo = gr.Video(value="./Video/spot.mp4",
                                     show_label=False,
                                     show_share_button=False,
                                     include_audio=False,
                                     loop=True,
                                     scale=2,
                                     interactive=False,
                                     autoplay=True,
                                     height=678,
                                     width=678,
                                     elem_classes="spot-video-container"
                                     )


            with gr.Column(scale=7):
                gr.HTML("""
                        <div style='
                        text-align: left;
                        font-family: system-ui, sans-serif;
                        font-size: 1.6em;
                        padding: 20px, 20px;
                        margin: 250px 0px 0px 0px;
                        border: none !important;
                            '>
                        Spot segmentation
                        </div>
                        """)
                gr.HTML("""
                        <div style='
                        text-align: left;
                        font-family: system-ui, sans-serif;
                        font-size: 1.1em;
                        line-height: 2;
                        padding: 20px, 20px;
                        margin: 10px 100px 0px 0px;
                        border: none !important;
                            '>
                        A deep learning-based approach is employed to perform instance segmentation on the spot signals within the fluorescence channel, enabling precise localization of the bounding box and mask for each spot.
                        </div>
                        """)

        with gr.Row():
            with gr.Column(scale=7):
                gr.HTML("""
                        <div style='
                        text-align: left;
                        font-family: system-ui, sans-serif;
                        font-size: 1.6em;
                        padding: 20px, 20px;
                        margin: 210px 0px 0px 100px;
                        border: none !important;
                            '>
                        Cell segmentation
                        </div>
                        """)
                gr.HTML("""
                        <div style='
                        text-align: left;
                        font-family: system-ui, sans-serif;
                        font-size: 1.1em;
                        line-height: 2;
                        padding: 20px, 20px;
                        margin: 10px 0px 0px 100px;
                        border: none !important;
                                 '>
                        The identical model is applied to conduct instance segmentation on the composite images derived from the bright-field, DAPI, and spot fluorescence channels. This approach precisely identifies cell boundaries, facilitating comprehensive analysis at the single-cell level.
                        </div>
                        """)


            with gr.Column(scale=10):
                cellVideo = gr.Video(value="./Video/cell.mp4",
                                     show_label=False,
                                     show_share_button=False,
                                     loop=True,
                                     scale=2,
                                     interactive=False,
                                     autoplay=True,
                                     height=678,
                                     width=678,
                                     elem_classes="cell-video-container"
                                     )



    with gr.Tab("Spot segmentation", elem_classes="tab-1"):
        with gr.Row():
            with gr.Column(scale=5):
                gr.HTML("""
                            <div style='
                                text-align: left;
                                font-family: system-ui, sans-serif;
                                font-size: 1.2em;
                                
                                padding: 20px 20px;
                                margin: 10px 0px 5px 10px;
                                display: inline-block;  /* 关键属性：宽度随内容扩展 */
                                background-color: #eef5fb;
                                padding: 4px 8px;       /* 背景内边距 */
                                border-radius: 4px;     /* 可选：圆角 */
                                
                            '>
                             ☄️ Image upload & experimental settings
                            
                            </div>
                        """)
                input_image_path = gr.Image(label="input image", type="filepath", show_share_button=False,
                                            show_download_button=False)


            with gr.Column(scale=2):
                pthDropdown = gr.Dropdown(
                    choices=["particlePth", "receptorPth", "vesiclePth", "smfishPth", "suntagPth", "spotData"],
                    label="select weight"
                )
                spotFactor = gr.Dropdown(
                    choices=["1", "2", "4", "8"],
                    label="select scale factor"
                )
                spotLoadExampleButton = gr.Button(value="Loading examples (optional)")
                process_button = gr.Button(value="Submit", elem_id="custom_button")
                channels = gr.Textbox(label="channels")
                valid_channels = gr.Textbox(label="valid channels")
                dtype = gr.Textbox(label="dtype")
                pixelMin = gr.Textbox(label="pixel value min")
                pixelMax = gr.Textbox(label="pixel value max")

                input_image_path.change(fn=ImageInfo, inputs=[input_image_path], outputs=[input_image_path, channels, valid_channels, dtype, pixelMin, pixelMax])

            with gr.Column(scale=4):
                gr.HTML("""
                            <div style='
                                text-align: left;
                                font-family: system-ui, sans-serif;
                                font-size: 1.2em;
    
                                padding: 20px 20px;
                                margin: 10px 0px 5px 10px;
                                display: inline-block;  /* 关键属性：宽度随内容扩展 */
                                background-color: #eef5fb;
                                padding: 4px 8px;       /* 背景内边距 */
                                border-radius: 4px;     /* 可选：圆角 */
                                
                            '>
                            ☄️ Quantitative results with spatial mapping
                            </div>
                        """)
                num_text = gr.Textbox(label="Spot number")
                with gr.Row():
                    output_bbox = gr.Image(label="Bbox", interactive=False)
                    output_mask = gr.Image(label="Mask", interactive=False)
                with gr.Row():
                    output_intensity = gr.Plot(label="Fluoescence intensity Distribution")
                    output_table = gr.DataFrame(label="Spatial Coordinate", interactive=False, height=300)




    with gr.Tab("Cell segmentation", elem_classes="tab-2"):

        with gr.Row():
            gr.HTML("""
                        <div style='
                            text-align: left;
                            font-family: system-ui, sans-serif;
                            font-size: 1.2em;

                            padding: 20px 20px;
                            margin: 10px 0px 5px 10px;
                            display: inline-block;  /* 关键属性：宽度随内容扩展 */
                            background-color: #eef5fb;
                            padding: 4px 8px;       /* 背景内边距 */
                            border-radius: 4px;     /* 可选：圆角 */
                        '>
                         ☄️ Image upload & experimental settings

                        </div>
                    """)

        with gr.Row():
            with gr.Column():
                input_cell_image_path = gr.Image(label="input image", type="filepath")

            input_cell_image_path.change(fn=imageShow, inputs=[input_cell_image_path], outputs=[input_cell_image_path])

            with gr.Column():
                cellPthDropdown = gr.Dropdown(
                    choices=["cellData"],
                    label="select weight"
                )
                cellLoadExampleButton = gr.Button(value="Loading examples (optional)")
                cell_process_button = gr.Button(value="Submit", elem_id="custom_button")

        with gr.Row():
            gr.HTML("""
                        <div style='
                            text-align: left;
                            font-family: system-ui, sans-serif;
                            font-size: 1.2em;

                            padding: 20px 20px;
                            margin: 10px 0px 5px 10px;
                            display: inline-block;  /* 关键属性：宽度随内容扩展 */
                            background-color: #eef5fb;
                            padding: 4px 8px;       /* 背景内边距 */
                            border-radius: 4px;     /* 可选：圆角 */
                        '>
                        ☄️ Quantitative results with spatial mapping
                        </div>
                    """)

        with gr.Row():
            cellNumber = gr.Textbox(label="Cell number")
        with gr.Row():
            with gr.Column():
                output_cell_bbox = gr.Image(label="Bbox", interactive=False)

            with gr.Column():
                output_cell_mask = gr.Image(label="Mask", interactive=False)



    with gr.Tab("Single-cell spot counting", elem_classes="tab-3"):
        with gr.Row():
            with gr.Column():
                gr.HTML("""<div style='text-align: left;font-family: system-ui, sans-serif;font-size: 1.2em;padding: 20px 20px;margin: 10px 0px 5px 10px;display: inline-block;background-color: #eef5fb;padding: 4px 8px;border-radius: 4px; 
                        '>☄️ Image upload & experimental settings </div>""")
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""<div style='text-align: center;font-family: system-ui, sans-serif;font-size: 1.0em;margin: 0px 0px 0px 0px;'> Input fluorescence channels of spots </div> """)
                        spotChannel_inputImagePath = gr.Image(label="spots image", type="filepath")
                        with gr.Row():
                            spotChannel_pth = gr.Dropdown(choices=["particlePth", "receptorPth", "vesiclePth", "smfishPth", "suntagPth", "spotData"],label="select weight")
                            spotChannel_scalefactor = gr.Dropdown(choices=["1", "2", "4", "8"],label="select scale factor",type="value")
                            spotChannel_inputImagePath.change(fn=imageShow, inputs=[spotChannel_inputImagePath],outputs=[spotChannel_inputImagePath])

                    with gr.Column():
                        gr.HTML("""<div style='text-align: center;font-family: system-ui, sans-serif;font-size: 1.0em;margin: 0px 0px 0px 0px;'> Input fluorescence channels of cells </div> """)
                        cellChannel_inputImagePath = gr.Image(label="cells image", type="filepath")
                        with gr.Row():
                            cellChannel_pth = gr.Dropdown(choices=["cellData"],label="select weight")
                            cellSpotLoadExampleButton = gr.Button(value="Loading examples (optional)")
                            cellChannel_inputImagePath.change(fn=imageShow, inputs=[cellChannel_inputImagePath], outputs=[cellChannel_inputImagePath])
                cellspot_process_button = gr.Button(value="Submit", elem_id="custom_button")

            with gr.Column():
                gr.HTML("""<div style='text-align: left;font-family: system-ui, sans-serif;font-size: 1.2em;padding: 20px 20px;margin: 10px 0px 5px 10px;display: inline-block; background-color: #eef5fb;padding: 4px 8px;  border-radius: 4px;'> ☄️ Quantitative results with spatial mapping</div>""")

                with gr.Row():
                    with gr.Column():
                        spotNumTextBox = gr.Textbox(label="spot number")
                    with gr.Column():
                        cellNumTextBox = gr.Textbox(label="cell number")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            spotBboxResult = gr.Image(label="Spot Bbox", interactive=False)
                            spotMaskResult = gr.Image(label="Spot Mask", interactive=False)
                    with gr.Column():
                        with gr.Row():
                            cellBboxResult = gr.Image(label="Cell Bbox", interactive=False)
                            cellMaskResult = gr.Image(label="Cell Mask", interactive=False)

                with gr.Row():
                    with gr.Column():
                        cellSpot_xiaotiqin = gr.Plot()
                    with gr.Column():
                        cstable = gr.DataFrame(label="Table", interactive=False,  height=250)




    with gr.Tab("Download"):
        for filename in list_files():
            gr.File(
                label=filename,
                value=os.path.join("./download", filename),
                visible=True,
                interactive=False
            )



    with gr.Tab("Help"):
        with gr.Row():
            gr.HTML("""
                        <div style='
                            text-align: left;
                            font-family: system-ui, sans-serif;
                            font-size: 1.2em;

                            padding: 20px 20px;
                            margin: 10px 0px 5px 10px;
                            display: inline-block;  /* 关键属性：宽度随内容扩展 */
                            background-color: #eef5fb;
                            padding: 4px 8px;       /* 背景内边距 */
                            border-radius: 4px;     /* 可选：圆角 */
                        '>
                         ☄️ System Monitoring

                        </div>
                    """)
        with gr.Row():
            nvitopShow = gr.HTML()
            # demo.load(run_nvidia_smi, None, nvitopShow)
        with gr.Row():
            gr.Button("Start nvidia-smi").click(
                run_nvidia_smi,
                outputs=nvitopShow,
                show_progress=False
            )

        with gr.Row():
            gr.HTML("""
                        <div style='
                            text-align: left;
                            font-family: system-ui, sans-serif;
                            font-size: 1.2em;

                            padding: 20px 20px;
                            margin: 10px 0px 5px 10px;
                            display: inline-block;  /* 关键属性：宽度随内容扩展 */
                            background-color: #eef5fb;
                            padding: 4px 8px;       /* 背景内边距 */
                            border-radius: 4px;     /* 可选：圆角 */
                        '>
                         ☄️ Parameter Configuration Guide

                        </div>
                    """)
        with gr.Row():
            gr.Markdown(markdown_content)




    process_button.click(fn=spotSeg, inputs=[input_image_path, pthDropdown, spotFactor], outputs=[output_bbox, output_mask, output_table, num_text, output_intensity])
    cell_process_button.click(fn=cellSeg, inputs=[input_cell_image_path, cellPthDropdown], outputs=[output_cell_bbox, output_cell_mask, cellNumber])
    cellspot_process_button.click(fn=cellSpotSeg,
                inputs=[spotChannel_inputImagePath, spotChannel_pth, spotChannel_scalefactor, cellChannel_inputImagePath, cellChannel_pth],
                outputs=[spotBboxResult, spotMaskResult, cellBboxResult, cellMaskResult, spotNumTextBox, cellNumTextBox, cstable, cellSpot_xiaotiqin]
                        )

    spotLoadExampleButton.click(fn=spotLoadexample, inputs=[], outputs=[input_image_path, pthDropdown, spotFactor])
    cellLoadExampleButton.click(fn=cellLoadExample, inputs=[], outputs=[input_cell_image_path, cellPthDropdown])
    cellSpotLoadExampleButton.click(fn=cellSpotLoadExample, inputs=[], outputs=[spotChannel_inputImagePath, spotChannel_pth, spotChannel_scalefactor, cellChannel_inputImagePath, cellChannel_pth])


if __name__ == "__main__":
    demo.launch(share=True)






