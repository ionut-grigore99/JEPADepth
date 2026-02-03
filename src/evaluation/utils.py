import os
import torch
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import PIL.Image as pil
import matplotlib

from src.inference.test import format_model_name
from src.utils import format_number, format_model_name

# Models which were trained with stereo supervision were trained with a nominal baseline of 0.1 units.
# The KITTI rig has a baseline of 54cm.
# Therefore, to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def compute_errors(gt, pred):
    """
        Computation of error metrics between predicted and ground truth depths.
    """
    if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
        thresh = torch.max((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
        rmse_log = torch.sqrt(rmse_log.mean())

        abs_rel = torch.mean(torch.abs(gt - pred) / gt)
 
        sq_rel = torch.mean((gt - pred) ** 2 / gt)
    else:
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def batch_post_process_disparity(l_disp, r_disp):
    """
        Apply the disparity post-processing method as introduced in Monodepthv1.
        "Unsupervised Monocular Depth Estimation With Left-Right Consistency" paper, pg 5-bottom right.
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def create_evaluation_html_report(dataset, mean_errors, conf, num_images, sample_images, num_parameters, avg_inference_time):
    """Create comprehensive HTML report for KITTI evaluation results
    
    Args:
        dataset: Name of the dataset (e.g., "kitti", "make3d", "cityscapes")
        mean_errors: Array of mean error values
        conf: Configuration dictionary
        num_images: Total number of evaluated images
        sample_images: List of dicts with keys: 'input_img', 'pred_depth', 'gt_depth', 'error_map' (all as numpy arrays)
        num_parameters: Total number of model parameters
        avg_inference_time: Average inference time per image in seconds
    """
    
    def numpy_to_base64(img_array, cmap='viridis'):
        """Convert numpy array to base64 encoded image"""
        if img_array.ndim == 2:  # Grayscale or depth map
            # Normalize to 0-255
            img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-6)
            img_normalized = (img_normalized * 255).astype(np.uint8)
            
            # Apply colormap
            cmap_func = matplotlib.colormaps.get_cmap(cmap)
            img_colored = (cmap_func(img_normalized)[:, :, :3] * 255).astype(np.uint8)
            img = pil.fromarray(img_colored)
        else:  # RGB image
            img = pil.fromarray(img_array.astype(np.uint8))
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
    
    def file_to_base64(file_path):
        """Convert image file to base64 encoded string"""
        try:
            with open(file_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            print(f"Warning: Could not load image {file_path}: {e}")
            return None
    
    error_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    
    # Dataset-specific configuration
    dataset_upper = dataset.upper()
    dataset_title = f"{dataset_upper} Depth Evaluation Report"
    
    # Load pipeline diagram based on dataset
    pipeline_diagram_filename = f"{dataset_upper} Evaluation Pipeline.png"
    pipeline_diagram_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                         'assets', pipeline_diagram_filename)
    pipeline_diagram_b64 = file_to_base64(pipeline_diagram_path)
    
    # Generate pipeline diagram HTML
    optional_steps_text = ""
    if dataset.lower() != "make3d":
        optional_steps_text = """
            The pipeline includes optional post-processing 
            (left-right consistency) and median scaling steps depending on the evaluation configuration.
        """

    pipeline_diagram_html = ""
    if pipeline_diagram_b64:
        pipeline_diagram_html = f"""
            <div class="pipeline-section">
                <h2>🔄 Evaluation Pipeline</h2>
                <div class="pipeline-diagram">
                    <img src="{pipeline_diagram_b64}" alt="{dataset_upper} Evaluation Pipeline" 
                        class="clickable-image" 
                        onclick="openLightbox(this.src, '{dataset_upper} Evaluation Pipeline')">
                    <p>
                        This diagram illustrates the complete {dataset_upper} evaluation pipeline, from model inference 
                        to final metric computation.
                        {optional_steps_text}
                        <br><em style="color: #667eea;">💡 Click on the image to view it in full size</em>
                    </p>
                </div>
            </div>
        """

    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{dataset_title}</title>
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
                max-width: 1200px;
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
                margin-bottom: 10px;
            }}
            
            .header .timestamp {{
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
                font-size: 1.8em;
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
            
            .metrics-section {{
                padding: 40px;
            }}
            
            .metrics-section h2 {{
                color: #667eea;
                margin-bottom: 30px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
                font-size: 1.8em;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            
            .metric-card {{
                background: #f8f9fa;
                padding: 25px;
                border-radius: 10px;
                border-left: 5px solid #667eea;
                transition: all 0.3s ease;
            }}
            
            .metric-card:hover {{
                transform: translateX(5px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            
            .metric-card .metric-name {{
                font-size: 1.1em;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .metric-card .metric-value {{
                font-size: 2.2em;
                font-weight: bold;
                color: #667eea;
                font-family: 'Courier New', monospace;
            }}
            
            .metric-card .metric-description {{
                font-size: 0.85em;
                color: #666;
                margin-top: 10px;
                line-height: 1.4;
            }}
            
            .metric-card.accuracy {{
                border-left-color: #10b981;
            }}
            
            .metric-card.accuracy .metric-value {{
                color: #10b981;
            }}
            
            .metric-card.error {{
                border-left-color: #ef4444;
            }}
            
            .metric-card.error .metric-value {{
                color: #ef4444;
            }}
            
            .comparison-table {{
                width: 100%;
                margin: 30px 0;
                border-collapse: collapse;
                background: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-radius: 10px;
                overflow: hidden;
            }}
            
            .comparison-table thead {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            
            .comparison-table th {{
                padding: 15px;
                text-align: center;
                font-weight: bold;
                font-size: 0.95em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .comparison-table td {{
                padding: 12px;
                text-align: center;
                border-bottom: 1px solid #e5e7eb;
            }}
            
            .comparison-table tbody tr:hover {{
                background: #f3f4f6;
            }}
            
            .comparison-table .model-row {{
                background: #fef3c7;
                font-weight: bold;
            }}
            
            .config-section {{
                padding: 30px 40px;
                background: #fff;
            }}
            
            .config-section h3 {{
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.4em;
            }}
            
            .config-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 12px;
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
            
            .info-box {{
                background: #dbeafe;
                border-left: 4px solid #3b82f6;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
            }}
            
            .info-box h4 {{
                color: #1e40af;
                margin-bottom: 10px;
                font-size: 1.1em;
            }}
            
            .info-box p {{
                color: #1e3a8a;
                line-height: 1.6;
                font-size: 0.95em;
            }}
            
            .info-box code {{
                background: #fff;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 0.9em;
                color: #ef4444;
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
            
            .prediction-gallery {{
                padding: 40px;
                background: #fff;
            }}
            
            .prediction-item {{
                background: #f8f9fa;
                border-radius: 15px;
                overflow: hidden;
                margin-bottom: 40px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            
            .prediction-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .prediction-header h4 {{
                font-size: 1.3em;
                margin: 0;
            }}
            
            .prediction-header .index {{
                background: rgba(255,255,255,0.2);
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
            }}
            
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
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
            
            .metric-formula {{
                background: #fff;
                padding: 12px 15px;
                border-radius: 8px;
                margin-top: 15px;
                border: 1px solid #e5e7eb;
                text-align: center;
            }}
            
            .formula-label {{
                font-weight: bold;
                color: #667eea;
                margin-bottom: 10px;
                display: block;
                font-size: 0.95em;
            }}
            
            .formula-image {{
                max-width: 100%;
                height: auto;
                cursor: zoom-in;
                transition: transform 0.2s ease;
                display: inline-block;
            }}
            
            .formula-image:hover {{
                transform: scale(1.05);
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
            
            .colorbar.hot {{
                background: linear-gradient(to right,
                    #0d0000, #4d0000, #8d0000, #cd0000, #ff0000,
                    #ff4d00, #ff8d00, #ffcd00, #ffff00, #ffffff);
            }}
            
            .colorbar-labels {{
                display: flex;
                justify-content: space-between;
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }}
            
            .pipeline-section {{
                padding: 40px;
                background: #fff;
            }}
            
            .pipeline-section h2 {{
                color: #667eea;
                margin-bottom: 30px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
                font-size: 1.8em;
            }}
            
            .pipeline-diagram {{
                background: #f8f9fa;
                border-radius: 15px;
                padding: 30px;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
            }}
            
            .pipeline-diagram img {{
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            }}
            
            .pipeline-diagram p {{
                margin-top: 20px;
                color: #666;
                font-size: 0.95em;
                line-height: 1.6;
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
                <h1>🎯 {dataset_title}</h1>
            </div>
            
            <!-- Summary Section -->
            <div class="summary">
                {summary_cards_html}
            </div>
            
            <!-- Results Table -->
            <div class="metrics-section">
                <h2 style="color: #667eea; margin-bottom: 20px;">📈 Evaluation Results</h2>
                {results_table_html}
            </div>
            
            <!-- Metrics Guide Section -->
            <div class="metrics-section">
                <h2>📖 Evaluation Metrics Guide</h2>
                
                <h3 style="color: #ef4444; margin-top: 30px; margin-bottom: 20px;">Error Metrics (Lower is Better)</h3>
                <div class="metrics-grid">
                    {error_metrics_html}
                </div>
                
                {accuracy_metrics_section_html}
            </div>
            
            <!-- Pipeline Diagram Section -->
            {pipeline_diagram_html}
            
            <!-- Info Boxes -->
            <div class="metrics-section">                
                <div class="info-box" style="background: #fef3c7; border-left-color: #f59e0b;">
                    <h4 style="color: #92400e;">⚠️ Median Scaling</h4>
                    <p style="color: #78350f;">
                        Self-supervised depth estimation is inherently <strong>scale-ambiguous</strong> because the network learns 
                        depth from monocular sequences without knowing the actual camera motion scale or baseline distance. 
                        This means predictions are correct in <em>relative proportions</em> but not in absolute metric units.
                        <br><br>
                        <strong>Median Scaling</strong> (Section 4.1 from <a href="https://arxiv.org/pdf/1704.07813" target="_blank">Godard et al., 2017</a>) aligns predicted depth to ground truth by computing:<br>
                        <code>scale_factor = median(GT_depth) / median(predicted_depth)</code><br>
                        Then scaling all predictions: <code>depth_scaled = scale_factor × depth_predicted</code>
                        <br><br>
                        This is a standard evaluation practice that allows fair comparison by correcting for the unknown scale, 
                        while still measuring the model's ability to estimate relative depth structure. For stereo-supervised models, 
                        a fixed scale factor of 5.4 is used instead (converting from training baseline 0.1m to KITTI baseline 0.54m).
                        <br><br>
                        <strong>Current setting:</strong> {median_scaling}
                    </p>
                </div>
                {post_processing_info_box_html}
            </div>
            
            <!-- Configuration Sections -->
            <div class="config-section">
                <h3>⚙️ Model Configuration</h3>
                <div class="config-grid">
                    {model_config_items_html}
                </div>
            </div>
            
            <div class="config-section" style="border-top: 2px solid #e5e7eb;">
                <h3>⚙️ Evaluation Configuration</h3>
                <div class="config-grid">
                    {eval_config_items_html}
                </div>
            </div>
            
            {sample_images_html}
            
            <!-- Footer -->
            <div class="footer">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate error metrics HTML with formulas
    error_metrics_html = ""
    error_metrics_info = {
        "abs_rel": {
            "name": "Absolute Relative Error",
            "description": "Measures the average relative difference between predicted and ground truth depth.",
            "image": "abs_rel.png"
        },
        "sq_rel": {
            "name": "Squared Relative Error",
            "description": "Penalizes larger errors more heavily by squaring the relative difference.",
            "image": "sq_rel.png"
        },
        "rmse": {
            "name": "Root Mean Squared Error",
            "description": "Measures the standard deviation of depth prediction errors in meters.",
            "image": "rmse.png"
        },
        "rmse_log": {
            "name": "RMSE in Log Space",
            "description": "More sensitive to relative errors at close range, invariant to scale.",
            "image": "rmse_log.png"
        }
    }
    
    # Get assets directory path
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    
    for i, name in enumerate(error_names[:4]):
        info = error_metrics_info[name]
        # Load formula image
        formula_img_path = os.path.join(assets_dir, info['image'])
        formula_img_b64 = file_to_base64(formula_img_path)
        
        if formula_img_b64:
            error_metrics_html += f"""
        <div class="metric-card error">
            <div class="metric-name">{info['name']}</div>
            <div class="metric-description">{info['description']}</div>
            <div class="metric-formula">
                <div class="formula-label">Formula:</div>
                <img src="{formula_img_b64}" alt="{name} formula" class="formula-image clickable-image" 
                     onclick="openLightbox(this.src, '{info['name']} Formula')">
            </div>
        </div>
        """
        else:
            # Fallback to text if image not found
            error_metrics_html += f"""
        <div class="metric-card error">
            <div class="metric-name">{info['name']}</div>
            <div class="metric-description">{info['description']}</div>
        </div>
        """
    
    # Generate accuracy metrics HTML with formulas (only for KITTI and Cityscapes, not Make3D)
    accuracy_metrics_html = ""
    accuracy_metrics_section_html = ""
    
    if dataset.lower() != "make3d":
        accuracy_metrics_info = {
            "a1": {
                "name": "Threshold Accuracy δ < 1.25",
                "description": "Percentage of pixels where the ratio between predicted and ground truth is less than 1.25.",
                "image": "a1.png"
            },
            "a2": {
                "name": "Threshold Accuracy δ < 1.25²",
                "description": "Percentage of pixels where the ratio between predicted and ground truth is less than 1.5625.",
                "image": "a2.png"
            },
            "a3": {
                "name": "Threshold Accuracy δ < 1.25³",
                "description": "Percentage of pixels where the ratio between predicted and ground truth is less than 1.9531.",
                "image": "a3.png"
            }
        }
        
        for i, name in enumerate(error_names[4:], start=4):
            info = accuracy_metrics_info[name]
            # Load formula image
            formula_img_path = os.path.join(assets_dir, info['image'])
            formula_img_b64 = file_to_base64(formula_img_path)
            
            if formula_img_b64:
                accuracy_metrics_html += f"""
        <div class="metric-card accuracy">
            <div class="metric-name">{info['name']}</div>
            <div class="metric-description">{info['description']}</div>
            <div class="metric-formula">
                <div class="formula-label">Formula:</div>
                <img src="{formula_img_b64}" alt="{name} formula" class="formula-image clickable-image" 
                     onclick="openLightbox(this.src, '{info['name']} Formula')">
            </div>
        </div>
        """
            else:
                # Fallback to text if image not found
                accuracy_metrics_html += f"""
        <div class="metric-card accuracy">
            <div class="metric-name">{info['name']}</div>
            <div class="metric-description">{info['description']}</div>
        </div>
        """
        
        # Create accuracy metrics section HTML
        accuracy_metrics_section_html = f"""
                <h3 style="color: #10b981; margin-top: 30px; margin-bottom: 20px;">Accuracy Metrics (Higher is Better)</h3>
                <div class="metrics-grid">
                    {accuracy_metrics_html}
                </div>
        """
    
    # Generate results row
    results_row = ""
    for error in mean_errors:
        results_row += f"<td><strong>{error:.3f}</strong></td>"
    
    # Generate results table based on dataset
    if dataset.lower() == "make3d":
        # Make3D: Only show error metrics (no accuracy metrics)
        results_table_html = f"""
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>abs_rel</th>
                            <th>sq_rel</th>
                            <th>rmse</th>
                            <th>rmse_log</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr class="model-row">
                            <td><strong>{format_model_name(conf['model_name'])}</strong></td>
                            <td><strong>{mean_errors[0]:.3f}</strong></td>
                            <td><strong>{mean_errors[1]:.3f}</strong></td>
                            <td><strong>{mean_errors[2]:.3f}</strong></td>
                            <td><strong>{mean_errors[3]:.3f}</strong></td>
                        </tr>
                    </tbody>
                </table>
        """
    else:
        # KITTI and Cityscapes: Show all 7 metrics
        results_table_html = f"""
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>abs_rel</th>
                            <th>sq_rel</th>
                            <th>rmse</th>
                            <th>rmse_log</th>
                            <th>δ < 1.25</th>
                            <th>δ < 1.25²</th>
                            <th>δ < 1.25³</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr class="model-row">
                            <td><strong>{format_model_name(conf['model_name'])}</strong></td>
                            {results_row}
                        </tr>
                    </tbody>
                </table>
        """
    
    # Determine scaling method
    if conf['evaluation_mode'] == 'stereo':
        scaling_method = f"Stereo scaling (factor={STEREO_SCALE_FACTOR})"
    elif conf['disable_median_scaling']:
        scaling_method = "No scaling applied"
    else:
        scaling_method = "Median scaling"
    
    # Generate post-processing info box (only for KITTI and Cityscapes, not Make3D)
    post_processing_info_box_html = ""
    if dataset.lower() != "make3d":
        post_processing_info_box_html = """
                <div class="info-box" style="background: #fef3c7; border-left-color: #f59e0b;">
                    <h4 style="color: #92400e;">⚠️ Post-Processing (Test-Time Augmentation)</h4>
                    <p style="color: #78350f;">
                        <strong>Post-processing</strong> is a test-time augmentation technique from the <a href="https://arxiv.org/pdf/1609.03677" target="_blank">Monodepth v1 paper</a> (pg. 5, bottom right) that improves depth predictions by leveraging left-right consistency.
                        <br><br>
                        <strong>How it works:</strong><br>
                        1. Run forward pass on original image → get disparity map <code>D_left</code><br>
                        2. Run forward pass on horizontally flipped image → get <code>D_right</code> (then flip back)<br>
                        3. Blend both disparities with distance-based weights:<br>
                        <code>D_final = w_left × D_left + w_right × D_right + w_center × (D_left + D_right)/2</code>
                        <br><br>
                        The weights favor <code>D_left</code> on the left side of the image, <code>D_right</code> on the right side, 
                        and average both in the center. This reduces edge artifacts and occlusion errors.
                        <br><br>
                        <strong>Trade-off:</strong> Requires 2× forward passes (slower inference) but typically improves 
                        <code>abs_rel</code>.
                        <br><br>
                        <strong>Current setting:</strong> {post_processing_status}
                    </p>
                </div>
        """
    
    # Generate model configuration items
    model_config_items_html = ""
    model_config = {}
    
    if conf['model_name'].startswith("pixio"):
        model_config = {
            "Model Name": format_model_name(conf['model_name']),
            "Number of Parameters": format_number(num_parameters),
            "Encoder": conf['pixio']['encoder'],
            "Pretrained Checkpoint": conf['pixio']['pretrained_ckp'] if conf['pixio']['pretrained_ckp'] else "None",
            "Weights Path": conf['pixio']['weights_path'],
            "Scales": str(conf['pixio']['scales']),
        }
    elif conf['model_name'] == "monodepth2":
        model_config = {
            "Model Name": format_model_name(conf['model_name']),
            "Number of Parameters": format_number(num_parameters),
            "ResNet Layers": str(conf['monodepth2']['num_layers']),
            "Pretrained": "Yes" if conf['monodepth2']['pretrained'] else "No",
            "Encoder Weights": conf['monodepth2']['encoder_weights_path'],
            "Decoder Weights": conf['monodepth2']['decoder_weights_path'],
            "Scales": str(conf['monodepth2']['scales']),
        }
    
    for key, value in model_config.items():
        model_config_items_html += f"""
        <div class="config-item">
            <div class="key">{key}</div>
            <div class="value">{value}</div>
        </div>
        """
    
    # Generate evaluation configuration items
    eval_config_items_html = ""
    eval_config_display = {}

    # Add Evaluation Split only for KITTI
    if dataset.lower() == "kitti":
        eval_config_display["Evaluation Split"] = conf['evaluation_split']

    # Add Evaluation Mode only for KITTI and Cityscapes
    if dataset.lower() in ["kitti", "cityscapes"]:
        eval_config_display["Evaluation Mode"] = conf['evaluation_mode']
        eval_config_display["Post-Processing"] = (
            "Enabled" if conf['evaluation_post_process'] else "Disabled"
        )
        eval_config_display["Median Scaling"] = (
            "Disabled" if conf['disable_median_scaling'] else "Enabled"
        )
        eval_config_display["Scaling Method"] = scaling_method

    # Add the rest of the fields
    eval_config_display.update({
        "Input Resolution": f"{conf['im_sz'][1]} × {conf['im_sz'][0]} (W × H)",
        "Min Depth": "0m" if dataset.lower() == "make3d" else f"{1e-3}m",
        "Max Depth": "70m" if dataset.lower() == "make3d" else "80m",
        "Number of Images": str(num_images),
        "Avg Inference Time": f"{avg_inference_time:.3f}s" if avg_inference_time > 0 else "N/A",
    })

    for key, value in eval_config_display.items():
        eval_config_items_html += f"""
        <div class="config-item">
            <div class="key">{key}</div>
            <div class="value">{value}</div>
        </div>
        """
    
    # Generate sample images HTML if provided
    sample_images_html = ""
    if sample_images and len(sample_images) > 0:
        sample_images_html = f"""
        <div class="prediction-gallery">
            <h2 style="color: #667eea; margin-bottom: 30px; padding-bottom: 10px; border-bottom: 2px solid #667eea;">
                🖼️ Sample Predictions (displayed {len(sample_images)} out of {num_images})
            </h2>
            
            <!-- Colorbar Legends -->
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
        """
        
        for idx, sample in enumerate(sample_images):
            input_img_b64 = numpy_to_base64(sample['input_img'])
            pred_depth_b64 = numpy_to_base64(sample['pred_depth'], cmap='viridis')
            gt_depth_b64 = numpy_to_base64(sample['gt_depth'], cmap='viridis')
            pred_disp_b64 = numpy_to_base64(sample['pred_disp'], cmap='plasma')
            
            sample_images_html += f"""
            <div class="prediction-item">
                <div class="prediction-header">
                    <h4>📸 {sample['filename']}</h4>
                    <div class="index">#{idx + 1}</div>
                </div>
                <div class="image-grid">
                    <div class="image-container">
                        <div class="image-label">Input Image</div>
                        <img src="{input_img_b64}" alt="Input Image" 
                             class="clickable-image" 
                             onclick="openLightbox(this.src, '{sample['filename']} - Input Image')">
                    </div>
                    <div class="image-container">
                        <div class="image-label">Predicted Depth</div>
                        <img src="{pred_depth_b64}" alt="Predicted Depth" 
                             class="clickable-image" 
                             onclick="openLightbox(this.src, '{sample['filename']} - Predicted Depth')">
                    </div>
                    <div class="image-container">
                        <div class="image-label">Ground Truth Depth</div>
                        <img src="{gt_depth_b64}" alt="Ground Truth Depth" 
                             class="clickable-image" 
                             onclick="openLightbox(this.src, '{sample['filename']} - Ground Truth Depth')">
                    </div>
                    <div class="image-container">
                        <div class="image-label">Predicted Disparity</div>
                        <img src="{pred_disp_b64}" alt="Predicted Disparity" 
                             class="clickable-image" 
                             onclick="openLightbox(this.src, '{sample['filename']} - Predicted Disparity')">
                    </div>
                </div>
            </div>
            """
        
        sample_images_html += "</div>"
    
    # Generate summary cards based on dataset
    if dataset.lower() == "make3d":
        # Make3D: Only 4 cards (Images, Resolution, Model, Device)
        summary_cards_html = f"""
                <div class="summary-card">
                    <div class="icon">📷</div>
                    <div class="value">{num_images}</div>
                    <div class="label">Images Evaluated</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🖼️</div>
                    <div class="value">{conf['im_sz'][1]}×{conf['im_sz'][0]}</div>
                    <div class="label">Input Resolution</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🤖</div>
                    <div class="value">{format_model_name(conf['model_name'])}</div>
                    <div class="label">Model</div>
                </div>
                <div class="summary-card">
                    <div class="icon">⚡</div>
                    <div class="value">{"CUDA" if torch.cuda.is_available() else "CPU"}</div>
                    <div class="label">Device</div>
                </div>
        """
    elif dataset.lower() == "cityscapes":
        # Cityscapes: 7 cards (all except Split)
        summary_cards_html = f"""
                <div class="summary-card">
                    <div class="icon">📷</div>
                    <div class="value">{num_images}</div>
                    <div class="label">Images Evaluated</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🖼️</div>
                    <div class="value">{conf['im_sz'][1]}×{conf['im_sz'][0]}</div>
                    <div class="label">Input Resolution</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🤖</div>
                    <div class="value">{format_model_name(conf['model_name'])}</div>
                    <div class="label">Model</div>
                </div>
                <div class="summary-card">
                    <div class="icon">⚡</div>
                    <div class="value">{"CUDA" if torch.cuda.is_available() else "CPU"}</div>
                    <div class="label">Device</div>
                </div>
                <div class="summary-card">
                    <div class="icon"></div>
                    <div class="value">{conf['evaluation_mode'].upper()}</div>
                    <div class="label">Evaluation Mode</div>
                </div>
                <div class="summary-card">
                    <div class="icon"></div>
                    <div class="value">{"Enabled" if conf['evaluation_post_process'] else "Disabled"}</div>
                    <div class="label">Post-Processing</div>
                </div>
                <div class="summary-card">
                    <div class="icon"></div>
                    <div class="value">{"Enabled" if not conf['disable_median_scaling'] else "Disabled"}</div>
                    <div class="label">Median Scaling</div>
                </div>
        """
    else:  # KITTI
        # KITTI: All 8 cards
        summary_cards_html = f"""
                <div class="summary-card">
                    <div class="icon">📷</div>
                    <div class="value">{num_images}</div>
                    <div class="label">Images Evaluated</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🖼️</div>
                    <div class="value">{conf['im_sz'][1]}×{conf['im_sz'][0]}</div>
                    <div class="label">Input Resolution</div>
                </div>
                <div class="summary-card">
                    <div class="icon">🤖</div>
                    <div class="value">{format_model_name(conf['model_name'])}</div>
                    <div class="label">Model</div>
                </div>
                <div class="summary-card">
                    <div class="icon">⚡</div>
                    <div class="value">{"CUDA" if torch.cuda.is_available() else "CPU"}</div>
                    <div class="label">Device</div>
                </div>
                <div class="summary-card">
                    <div class="icon"></div>
                    <div class="value">{conf['evaluation_split'].upper()}</div>
                    <div class="label">Evaluation Split</div>
                </div>
                <div class="summary-card">
                    <div class="icon"></div>
                    <div class="value">{conf['evaluation_mode'].upper()}</div>
                    <div class="label">Evaluation Mode</div>
                </div>
                <div class="summary-card">
                    <div class="icon"></div>
                    <div class="value">{"Enabled" if conf['evaluation_post_process'] else "Disabled"}</div>
                    <div class="label">Post-Processing</div>
                </div>
                <div class="summary-card">
                    <div class="icon"></div>
                    <div class="value">{"Enabled" if not conf['disable_median_scaling'] else "Disabled"}</div>
                    <div class="label">Median Scaling</div>
                </div>
        """
    
    # Fill template
    html_content = html_template.format(
        dataset_title=dataset_title,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_images=num_images,
        input_resolution=f"{conf['im_sz'][1]}×{conf['im_sz'][0]}",
        model_name=format_model_name(conf['model_name']),
        device="CUDA" if torch.cuda.is_available() else "CPU",
        split=conf['evaluation_split'].upper(),
        mode=conf['evaluation_mode'].upper(),
        pipeline_diagram_html=pipeline_diagram_html,
        error_metrics_html=error_metrics_html,
        accuracy_metrics_section_html=accuracy_metrics_section_html,
        results_table_html=results_table_html,
        results_row=results_row,
        median_scaling = "Enabled" if not conf['disable_median_scaling'] or dataset.lower() == 'make3d' else "Disabled",
        post_processing = "Enabled" if conf['evaluation_post_process'] else "Disabled",
        scaling_method=scaling_method,
        post_processing_status="Enabled (2× inference cost)" if conf['evaluation_post_process'] else "Disabled (1× inference cost)",
        summary_cards_html=summary_cards_html,
        post_processing_info_box_html=post_processing_info_box_html,
        model_config_items_html=model_config_items_html,
        eval_config_items_html=eval_config_items_html,
        sample_images_html=sample_images_html
    )
    
    # Save HTML report
    html_directory = conf['htmls_path']
    os.makedirs(html_directory, exist_ok=True)
    
    model_name_clean = conf['model_name'].replace('/', '_')
    report_filename = f"{dataset.lower()}{'_' + conf['evaluation_split'] if dataset.lower() == 'kitti' else ''}_evaluation_{model_name_clean}.html"
    report_path = os.path.join(html_directory, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n-> HTML Evaluation Report saved to: {report_path}")
    return report_path