import os
import glob
import numpy as np
import lovely_tensors as lt
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
import base64
from datetime import datetime
from io import BytesIO
from torchvision import transforms

from src.models.pixio.dpt import DPTDepth
from src.models.monodepth2.monodepth2 import MonoDepth2
from src.utils import disp_to_depth, count_parameters, format_number, image_to_base64, numpy_to_base64, format_model_name
from src.config.conf import Conf
import time

def create_html_report(results, html_directory, conf, num_parameters=0, avg_inference_time=0.0):
    """Create a comprehensive HTML report with all predictions"""
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Monocular Depth Estimation Inference Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}
            
            .header .subtitle {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .header .timestamp {{
                margin-top: 15px;
                font-size: 0.9em;
                opacity: 0.8;
            }}
            
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f8f9fa;
                border-bottom: 3px solid #667eea;
            }}
            
            .summary-card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            
            .summary-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 12px rgba(0,0,0,0.15);
            }}
            
            .summary-card .icon {{
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            
            .summary-card .value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 10px 0;
            }}
            
            .summary-card .label {{
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .config-section {{
                padding: 30px;
                background: #fff;
            }}
            
            .config-section h2 {{
                color: #667eea;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
            }}
            
            .config-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 15px;
                margin-top: 20px;
            }}
            
            .config-item {{
                background: #f8f9fa;
                padding: 12px 15px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .config-item .key {{
                font-weight: bold;
                color: #333;
                font-size: 0.95em;
            }}
            
            .config-item .value {{
                color: #666;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }}
            
            .explanation-container {{
                display: grid;
                gap: 20px;
                padding: 20px 30px;
            }}
            
            .explanation-item {{
                display: flex;
                gap: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 4px solid #667eea;
                transition: all 0.3s ease;
            }}
            
            .explanation-item:hover {{
                background: #e8ecf3;
                transform: translateX(5px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            
            .explanation-icon {{
                font-size: 2em;
                min-width: 50px;
                text-align: center;
            }}
            
            .explanation-content {{
                flex: 1;
            }}
            
            .explanation-content h4 {{
                margin: 0 0 10px 0;
                color: #2c3e50;
                font-size: 1.1em;
            }}
            
            .explanation-content p {{
                margin: 0;
                color: #555;
                line-height: 1.6;
            }}
            
            .explanation-content code {{
                background: #e1e8ed;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                color: #d63384;
            }}
            
            .pipeline-note {{
                margin-top: 10px;
                padding: 15px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 8px;
                text-align: center;
                font-size: 0.95em;
                line-height: 1.8;
            }}
            
            .pipeline-note code {{
                background: rgba(255,255,255,0.2);
                color: #fde724;
                padding: 2px 6px;
                border-radius: 3px;
            }}
            
            .results-section {{
                padding: 30px;
            }}
            
            .results-section h2 {{
                color: #667eea;
                margin-bottom: 30px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
            }}
            
            .result-item {{
                margin-bottom: 50px;
                background: #f8f9fa;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            .result-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .result-header h3 {{
                font-size: 1.3em;
                margin: 0;
            }}
            
            .result-header .index {{
                background: rgba(255,255,255,0.2);
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
            }}
            
            .result-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                padding: 20px 30px;
                background: white;
            }}
            
            .stat-box {{
                text-align: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            
            .stat-box .stat-label {{
                color: #666;
                font-size: 0.85em;
                margin-bottom: 5px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .stat-box .stat-value {{
                color: #667eea;
                font-size: 1.3em;
                font-weight: bold;
            }}
            
            .result-images {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                padding: 30px;
                background: white;
            }}
            
            .image-container {{
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }}
            
            .image-container:hover {{
                transform: scale(1.02);
                box-shadow: 0 8px 12px rgba(0,0,0,0.2);
            }}
            
            .image-container img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            
            .image-label {{
                background: #667eea;
                color: white;
                padding: 12px;
                font-weight: bold;
                font-size: 1em;
                text-align: center;
            }}
            
            .depth-comparison {{
                padding: 30px;
                background: white;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            
            .depth-viz {{
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            .depth-viz img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            
            .depth-viz .label {{
                background: #667eea;
                color: white;
                padding: 10px;
                text-align: center;
                font-weight: bold;
            }}
            
            .colorbar-legend {{
                padding: 20px 30px;
                background: #f8f9fa;
                margin: 20px 30px;
                border-radius: 10px;
                text-align: center;
            }}
            
            .colorbar-legend h4 {{
                margin-bottom: 15px;
                color: #333;
            }}
            
            .colorbar {{
                height: 30px;
                border-radius: 5px;
                margin: 10px 0;
            }}
            
            .colorbar.viridis {{
                background: linear-gradient(to right, 
                    #440154, #482777, #3e4989, #31688e, #26828e,
                    #1f9e89, #35b779, #6ece58, #b5de2b, #fde724);
            }}
            
            .colorbar.plasma {{
                background: linear-gradient(to right, 
                    #0d0887, #5302a3, #8b0aa5, #b93289, #db5c68,
                    #f48849, #febd2a, #f0f921);
            }}
            
            .colorbar-labels {{
                display: flex;
                justify-content: space-between;
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }}
            
            .footer {{
                background: #2c3e50;
                color: white;
                text-align: center;
                padding: 30px;
                font-size: 0.9em;
            }}
            
            .footer a {{
                color: #667eea;
                text-decoration: none;
            }}
            
            .footer a:hover {{
                text-decoration: underline;
            }}
            
            @media (max-width: 768px) {{
                .summary {{
                    grid-template-columns: 1fr;
                }}
                
                .result-images {{
                    grid-template-columns: 1fr;
                }}
                
                .depth-comparison {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            .progress-bar {{
                width: 100%;
                height: 4px;
                background: #e0e0e0;
                position: relative;
                overflow: hidden;
            }}
            
            .progress-bar::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                width: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
            }}
            
            /* Lightbox Modal Styles */
            .lightbox-modal {{
                display: none;
                position: fixed;
                z-index: 9999;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.95);
                cursor: zoom-out;
                animation: fadeIn 0.3s ease;
            }}
            
            .lightbox-modal.active {{
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            
            .lightbox-content {{
                max-width: 95%;
                max-height: 95%;
                object-fit: contain;
                box-shadow: 0 0 50px rgba(255, 255, 255, 0.2);
                animation: zoomIn 0.3s ease;
            }}
            
            .lightbox-close {{
                position: absolute;
                top: 20px;
                right: 40px;
                color: white;
                font-size: 50px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                z-index: 10000;
            }}
            
            .lightbox-close:hover {{
                color: #667eea;
                transform: scale(1.1);
            }}
            
            .lightbox-caption {{
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                color: white;
                background: rgba(0, 0, 0, 0.7);
                padding: 15px 30px;
                border-radius: 10px;
                font-size: 1.1em;
                max-width: 80%;
                text-align: center;
            }}
            
            .clickable-image {{
                cursor: zoom-in;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            
            .clickable-image:hover {{
                transform: scale(1.02);
                box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            
            @keyframes zoomIn {{
                from {{ transform: scale(0.8); opacity: 0; }}
                to {{ transform: scale(1); opacity: 1; }}
            }}
        </style>
    </head>
    <body>
        <!-- Lightbox Modal -->
        <div id="lightbox" class="lightbox-modal" onclick="closeLightbox()">
            <span class="lightbox-close">&times;</span>
            <img class="lightbox-content" id="lightbox-img">
            <div class="lightbox-caption" id="lightbox-caption"></div>
        </div>
        
        <script>
            function openLightbox(imgSrc, caption) {{
                const lightbox = document.getElementById('lightbox');
                const lightboxImg = document.getElementById('lightbox-img');
                const lightboxCaption = document.getElementById('lightbox-caption');
                
                lightbox.classList.add('active');
                lightboxImg.src = imgSrc;
                lightboxCaption.textContent = caption;
                document.body.style.overflow = 'hidden';
            }}
            
            function closeLightbox() {{
                const lightbox = document.getElementById('lightbox');
                lightbox.classList.remove('active');
                document.body.style.overflow = 'auto';
            }}
            
            // Close lightbox on Escape key
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'Escape') {{
                    closeLightbox();
                }}
            }});
            
            // Prevent closing when clicking on the image itself
            document.getElementById('lightbox-img').addEventListener('click', function(event) {{
                event.stopPropagation();
            }});
        </script>
        
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1>🎯 Monocular Depth Estimation Inference Report</h1>
            </div>
            
            <div class="progress-bar"></div>
            
            <!-- Summary Section -->
            <div class="summary">
                <div class="summary-card">
                    <div class="icon">📷</div>
                    <div class="value">{num_images}</div>
                    <div class="label">Images Processed</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🖼️</div>
                    <div class="value">{input_resolution}</div>
                    <div class="label">Input Resolution</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🤖</div>
                    <div class="value">{model_name}</div>
                    <div class="label">Model Architecture</div>
                </div>
                <div class="summary-card">
                    <div class="icon">⚡</div>
                    <div class="value">{device}</div>
                    <div class="label">Device</div>
                </div>
            </div>
            
            <!-- Configuration Section -->
            <div class="config-section">
                <h2>⚙️ Model Configuration</h2>
                <div class="config-grid">
                    {config_items}
                </div>
            </div>
            
            <!-- Explanation Section -->
            <div class="config-section">
                <h2>📖 Output Visualization Guide</h2>
                <div class="explanation-container">
                    <div class="explanation-item">
                        <div class="explanation-icon"></div>
                        <div class="explanation-content">
                            <h4>Input Image</h4>
                            <p>The input RGB image resized to dimensions on which we want to perform inference ({input_height}×{input_width}). 
                            This is the image fed into the depth estimation network.</p>
                        </div>
                    </div>
                    
                    <div class="explanation-item">
                        <div class="explanation-icon"></div>
                        <div class="explanation-content">
                            <h4>Normalized Disparity</h4>
                            <p>Direct sigmoid-activated output from the depth decoder in range [0, 1]. 
                            This is a <strong>normalized disparity representation</strong>, not actual disparity.</p>
                        </div>
                    </div>

                    <div class="explanation-item">
                        <div class="explanation-icon"></div>
                        <div class="explanation-content">
                            <h4>Disparity (Scale-Ambiguous)</h4>
                            <p>Scaled version of the network output using:<br>
                            <code>disparity = min_disp + (max_disp - min_disp) × sigmoid_output</code> where <code>min_disp = 1/{max_depth}</code> and <code>max_disp = 1/{min_depth}</code>.<br>
                            ⚠️ These units are <strong>scale-ambiguous</strong> since the network has no knowledge of metric scale!</p>
                        </div>
                    </div>

                    <div class="explanation-item">
                        <div class="explanation-icon"></div>
                        <div class="explanation-content">
                            <h4>Depth (Scale-Ambiguous)</h4>
                            <p>Computed as: <code>depth = 1 / disparity</code><br>
                            ⚠️ The predicted depth is <strong>not metric</strong>. Self-supervised depth estimation is inherently <em>scale-ambiguous</em> because the network has no access to real-world distances.<br>
                            To get metric depth, rescale the prediction using the stereo baseline correction:<br>
                            <code>depth_metric = scale_factor × depth</code> where <code>scale_factor = 5.4</code> converts from the nominal training baseline (0.1) to the real KITTI baseline (0.54 m).</p>
                        </div>
                    </div>
                    
                    <div class="pipeline-note">
                        <strong>⚙️ Processing Pipeline:</strong><br>
                        Input Image → Depth Model (with <strong>Sigmoid</strong> activation) → Normalized Disparity [0,1] → <code>disp = (1/{max_depth}) + ((1/{min_depth}) - (1/{max_depth})) × norm_disp</code> → <code>depth = 1 / disp</code>
                    </div>
                </div>
            </div>
            
            <!-- Results Section -->
            <div class="results-section">
                <h2>🖼️ Sample Predictions (average inference time: {avg_inference_time})</h2>
                
                <!-- Colorbar Legend -->
                <div class="colorbar-legend">
                    <h4>Depth Colormap Legend (Viridis)</h4>
                    <div class="colorbar viridis"></div>
                    <div class="colorbar-labels">
                        <span>Low Depth (Near)</span>
                        <span>High Depth (Far)</span>
                    </div>
                    <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
                        Warmer colors (yellow/green) represent farther objects, cooler colors (purple/blue) represent closer objects
                    </p>
                </div>

                <!-- Colorbar Legend -->
                <div class="colorbar-legend">
                    <h4>Disparity Colormap Legend (Plasma)</h4>
                    <div class="colorbar plasma"></div>
                    <div class="colorbar-labels">
                        <span>Low Disparity (Far)</span>
                        <span>High Disparity (Near)</span>
                    </div>
                    <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
                        Higher disparity values (yellow/orange) represent closer objects, lower values (purple/blue) represent farther objects
                    </p>
                </div>
                
                {results_html}
            </div>
            
            <!-- Footer -->
            <div class="footer">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate config items HTML
    config_items_html = ""
    
    # Build config display based on model type
    config_display = {
        "Model Name": format_model_name(conf['model_name']),
        "Number of parameters": format_number(num_parameters),
    }
    
    # Add model-specific configuration
    if conf['model_name'].startswith("pixio"):
        config_display.update({
            "Encoder": conf['pixio']['encoder'],
            "Pretrained Checkpoint": conf['pixio'].get('pretrained_ckp', 'None'),
            "Weights Path": conf['pixio'].get('weights_path', 'None'),
            "Scales": str(conf['pixio']['scales']),
        })
    elif conf['model_name'] == "monodepth2":
        config_display.update({
            "ResNet Layers": conf['monodepth2']['num_layers'],
            "Pretrained": "Yes" if conf['monodepth2']['pretrained'] else "No",
            "Encoder Weights Path": conf['monodepth2']['encoder_weights_path'],
            "Decoder Weights Path": conf['monodepth2']['decoder_weights_path'],
            "Scales": str(conf['monodepth2']['scales']),
        })
    
    for key, value in config_display.items():
        config_items_html += f"""
        <div class="config-item">
            <div class="key">{key}</div>
            <div class="value">{value}</div>
        </div>
        """
    
    # Generate results HTML
    results_html = ""
    for idx, result in enumerate(results):
        filename = result['filename']
        result_html = f"""
        <div class="result-item">
            <div class="result-header">
                <h3>📸 {filename}</h3>
                <div class="index">#{idx + 1}</div>
            </div>
            
            <div class="result-stats">
                <div class="stat-box">
                    <div class="stat-label">Original Size</div>
                    <div class="stat-value">{result['original_size'][0]} × {result['original_size'][1]}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Model Input</div>
                    <div class="stat-value">{result['model_input_size'][0]} × {result['model_input_size'][1]}</div>
                </div>
            </div>
            
            <div class="result-images">
                <div class="image-container">
                    <div class="image-label">Input Image</div>
                    <img src="{result['input_image_b64']}" alt="Input Image" 
                         class="clickable-image" 
                         onclick="openLightbox(this.src, '{filename} - Input Image')">
                </div>
                <div class="image-container">
                    <div class="image-label">Depth Prediction</div>
                    <img src="{result['depth_b64']}" alt="Depth" 
                         class="clickable-image" 
                         onclick="openLightbox(this.src, '{filename} - Depth Prediction')">
                </div>
            </div>
            
            <div class="depth-comparison">
                <div class="image-container">
                    <div class="image-label">Disparity Prediction</div>
                    <img src="{result['disparity_b64']}" alt="Disparity" 
                         class="clickable-image" 
                         onclick="openLightbox(this.src, '{filename} - Disparity Prediction')">
                </div>
                <div class="image-container">
                    <div class="image-label">Normalized Disparity Prediction</div>
                    <img src="{result['normalized_disparity_b64']}" alt="Normalized Disparity" 
                         class="clickable-image" 
                         onclick="openLightbox(this.src, '{filename} - Normalized Disparity Prediction')">
                </div>
            </div>
        </div>
        """
        results_html += result_html
    
    # Fill template
    html_content = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_images=len(results),
        input_resolution=f"{conf['im_sz'][1]}×{conf['im_sz'][0]}",
        model_name=format_model_name(conf['model_name']),
        device="CUDA" if torch.cuda.is_available() else "CPU",
        num_parameters=format_number(num_parameters),
        avg_inference_time=f"{avg_inference_time:.3f}s" if avg_inference_time > 0 else "N/A",
        config_items=config_items_html,
        results_html=results_html,
        input_height=conf['im_sz'][0],
        input_width=conf['im_sz'][1],
        min_depth=conf.get('min_depth', 0.1),
        max_depth=conf.get('max_depth', 100)
    )
    
    # Create HTML directory if it doesn't exist
    os.makedirs(html_directory, exist_ok=True)
    
    # Save HTML report with model name suffix
    model_name_clean = conf['model_name'].replace('/', '_')
    report_filename = f"inference_report_{model_name_clean}.html"
    report_path = os.path.join(html_directory, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   HTML Report saved to: {report_path}")
    return report_path

def test_simple(conf):
    """
        Function to predict depth map(s) for a single image or folder of images as input.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_input_height = conf['im_sz'][0]
    model_input_width = conf['im_sz'][1]

    # Preparing model
    if conf['model_name'].startswith("pixio"):
        model = DPTDepth(conf['pixio']['encoder'], conf['pixio']['pretrained_ckp'], conf['pixio']['scales'])
        model.from_pretrained(weights_path=conf['pixio']['weights_path'], device=device)
    elif conf['model_name'] == "monodepth2":
        model = MonoDepth2(num_layers=conf['monodepth2']['num_layers'], pretrained=conf['monodepth2']['pretrained'], scales=conf['monodepth2']['scales'])
        model.from_pretrained(encoder_weights_path=conf['monodepth2']['encoder_weights_path'], decoder_weights_path=conf['monodepth2']['decoder_weights_path'], device=device)
    else:
        raise NotImplementedError("Model not implemented for evaluation!")
    model.to(device)
    model.eval()
    
    # Count model parameters
    total_params = count_parameters(model)

    # Setup directories
    images_output_directory = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(images_output_directory, exist_ok=True)
    
    # HTML reports directory from config
    html_directory = conf.get('htmls_path', 'htmls')
    
    # Finding input image(s)
    if os.path.isfile(conf['image_path_inference']):
        # Testing on a single image
        paths = [conf['image_path_inference']]
    elif os.path.isdir(conf['image_path_inference']):
        # Testing on a folder of images
        paths = glob.glob(os.path.join(conf['image_path_inference'], "*"))
    else:
        raise Exception("Can not find conf['image_path_inference']: {}".format(conf['image_path_inference']))

    print("\n-> Predicting on {:d} test image(s)".format(len(paths)))

    # Store results for HTML report
    results = []
    inference_times = []

    # Predicting on each image
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image_resized = input_image.resize((model_input_width, model_input_height), pil.Resampling.LANCZOS)
            input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0)

            # Prediction (normalized disparity, disparity, depth) with timing
            input_tensor = input_tensor.to(device)
            
            # Measure inference time
            start_time = time.time()
            normalized_disparity = model(input_tensor)['disp', 0]
            disparity, depth = disp_to_depth(normalized_disparity, conf['min_depth'], conf['max_depth'])
            
            # Synchronize CUDA if using GPU to get accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Resize outputs to original image size
            normalized_disparity_resized = torch.nn.functional.interpolate(normalized_disparity, (original_height, original_width), mode="bicubic", align_corners=True)
            disparity_resized = torch.nn.functional.interpolate(disparity, (original_height, original_width), mode="bicubic", align_corners=True)
            depth_resized = torch.nn.functional.interpolate(depth, (original_height, original_width), mode="bicubic", align_corners=True)

            # Convert outputs to numpy
            normalized_disparity_resized_np = normalized_disparity_resized.squeeze().cpu().numpy()
            disparity_resized_np = disparity_resized.squeeze().cpu().numpy()
            depth_resized_np = depth_resized.squeeze().cpu().numpy()

            # Create visualizations for normalized disparity, disparity and depth
            normalizer = mpl.colors.Normalize(vmin=normalized_disparity_resized_np.min(), vmax=normalized_disparity_resized_np.max())
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
            normalized_disparity_viz = (mapper.to_rgba(normalized_disparity_resized_np)[:, :, :3] * 255).astype(np.uint8)

            vmax = np.percentile(disparity_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disparity_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
            disparity_viz = (mapper.to_rgba(disparity_resized_np)[:, :, :3] * 255).astype(np.uint8)

            vmax = np.percentile(depth_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=depth_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
            depth_viz = (mapper.to_rgba(depth_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(depth_viz)
            im.save(image_path.replace("input_images", "output_images").replace(".jpg", "_depth.jpg"))

            # Store result data for HTML report
            result_data = {
                'filename': os.path.basename(image_path),
                'original_size': (original_width, original_height),
                'model_input_size': (model_input_width, model_input_height),
                'input_image_b64': image_to_base64(input_image_resized),
                'depth_b64': numpy_to_base64(depth_viz),
                'disparity_b64': numpy_to_base64(disparity_viz),
                'normalized_disparity_b64': numpy_to_base64(normalized_disparity_viz),
            }
            results.append(result_data)

            print("   Processed {:d} of {:d} images - saved to: {}".format(idx + 1, len(paths), images_output_directory))

    # Calculate average inference time
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    print(f"\n-> Average inference time: {avg_inference_time:.3f}s per image")

    # Generate HTML report
    print("\n-> Generating HTML report...")
    create_html_report(results, html_directory, conf, num_parameters=total_params, avg_inference_time=avg_inference_time)

    print('\n-> Done!')

if __name__ == '__main__':

    lt.monkey_patch()

    conf = Conf().conf
    test_simple(conf)

