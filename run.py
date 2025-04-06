import torch
import argparse
from reconstruction.style_recon import StyleReconstructor
from reconstruction.content_recon import ContentReconstructor
from utils.image_utils import preprocess_image, save_image
from models.vgg_fast import VggFeatureExtractor
from neural_style_transfer import NeuralStyleTransfer
from visualization.visualize import feature_visualization

def run_content_reconstruction(args):
    """Run content reconstruction with given arguments"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running content reconstruction on {device} with {args.model_type}")
    
    reconstructor = ContentReconstructor(
        image_path=args.content_image, 
        target_layer=args.target_layer, 
        device=device,
        model_type=args.model_type
    )
    
    generated_img, loss_hist = reconstructor.reconstruct(
        iterations=args.iterations,
        learning_rate=args.lr,
        use_content_init=args.use_content_init,
        optimizer_type=args.optimizer,
        lbfgs_max_iter=20,
        tv_weight=args.tv_weight,
        noise_scale=args.noise_scale
    )
    
    save_image(generated_img, args.output)
    print(f"Content reconstruction saved to {args.output}")

def run_style_reconstruction(args):
    """Run style reconstruction with given arguments"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running style reconstruction on {device} with {args.model_type}")
    
    reconstructor = StyleReconstructor(
        style_image_path=args.style_image,
        device=device,
        model_type=args.model_type,
        image_size=args.image_size
    )
    
    reconstructed_image = reconstructor.reconstruct(
        optimizer_type=args.optimizer,
        init_method=args.init_method,
        num_steps=args.iterations,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        lr=args.lr,
        noise_factor=args.noise_scale,
        print_freq=10,
        show_images=args.show_images
    )
    
    save_image(reconstructed_image, args.output)
    print(f"Style reconstruction saved to {args.output}")

def run_feature_visualization(args):
    """Run feature visualization with given arguments"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running feature visualization on {device} with {args.model_type}")
    
    tensor = preprocess_image(args.image, size=args.image_size)
    model = VggFeatureExtractor(args.target_layer, device=device, model_type=args.model_type)
    features = model(tensor)
    feature_visualization(feature_maps=features, layer_name = f"Layer {args.target_layer}" , save = args.save)

def run_neural_style_transfer(args):
    """Run neural style transfer with given arguments"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running neural style transfer on {device} with {args.model_type}")
    
    nst = NeuralStyleTransfer(
        content_image_path=args.content_image,
        style_image_path=args.style_image,
        content_layer=args.content_layer,
        style_layers=args.style_layers,
        device=device,
        model_type=args.model_type,
        image_size=args.image_size
    )
    
    stylized_image = nst.transfer(
        optimizer_type=args.optimizer,
        init_method=args.init_method,
        num_steps=args.iterations,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        lr=args.lr,
        show_images=args.show_images,
        output_path=args.output_dir
    )
    
    print(f"Neural style transfer complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Neural Style Transfer and Feature Visualization Tool')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Common arguments for all modes
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--model_type', type=str, default='vgg19', choices=['vgg16', 'vgg19'], help='VGG model type')
    # Removed the --use_gpu flag as we're now auto-detecting CUDA availability
    common_parser.add_argument('--image_size', type=int, default=512, help='Image size for processing')
    common_parser.add_argument('--iterations', type=int, default=100, help='Number of optimization iterations')
    common_parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    common_parser.add_argument('--optimizer', type=str, default='lbfgs', choices=['adam', 'lbfgs'], help='Optimizer type')
    common_parser.add_argument('--tv_weight', type=float, default=1e-3, help='Total variation weight')
    common_parser.add_argument('--noise_scale', type=float, default=0.1, help='Noise scale for initialization')
    common_parser.add_argument('--show_images', action='store_true', help='Show images during optimization')
    
    # Content reconstruction specific arguments
    content_parser = subparsers.add_parser('content', parents=[common_parser], help='Content reconstruction')
    content_parser.add_argument('--content_image', type=str, required=True, help='Path to content image')
    content_parser.add_argument('--target_layer', type=int, default=20, help='Target layer for content reconstruction')
    content_parser.add_argument('--use_content_init', action='store_true', help='Use content image as initialization')
    content_parser.add_argument('--output', type=str, default='content_reconstructed.jpg', help='Output file path')
    
    # Style reconstruction specific arguments
    style_parser = subparsers.add_parser('style', parents=[common_parser], help='Style reconstruction')
    style_parser.add_argument('--style_image', type=str, required=True, help='Path to style image')
    style_parser.add_argument('--style_weight', type=float, default=1e3, help='Style weight')
    style_parser.add_argument('--init_method', type=str, default='noise', choices=['noise', 'style', 'style_with_noise'], help='Initialization method')
    style_parser.add_argument('--output', type=str, default='style_reconstructed.jpg', help='Output file path')
    
    # Feature visualization specific arguments
    vis_parser = subparsers.add_parser('visualize', parents=[common_parser], help='Feature visualization')
    vis_parser.add_argument('--image', type=str, required=True, help='Path to image')
    vis_parser.add_argument('--target_layer', type=int, default=10, help='Target layer for visualization')
    vis_parser.add_argument('--save', action="store_true", help='to save all the feature maps')
    
    # Neural Style Transfer specific arguments
    nst_parser = subparsers.add_parser('nst', parents=[common_parser], help='Neural Style Transfer')
    nst_parser.add_argument('--content_image', type=str, required=True, help='Path to content image')
    nst_parser.add_argument('--style_image', type=str, required=True, help='Path to style image')
    nst_parser.add_argument('--content_layer', type=int, default=21, help='Content layer')
    nst_parser.add_argument('--style_layers', type=int, nargs='+', default=[1, 6, 11, 20, 29], help='Style layers')
    nst_parser.add_argument('--content_weight', type=float, default=1, help='Content weight')
    nst_parser.add_argument('--style_weight', type=float, default=1e4, help='Style weight')
    nst_parser.add_argument('--init_method', type=str, default='content', choices=['noise', 'content', 'content_with_noise', 'style', 'style_with_noise'], help='Initialization method')
    nst_parser.add_argument('--output_dir', type=str, default='output/style_transfer', help='Output directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the selected mode
    if args.mode == 'content':
        run_content_reconstruction(args)
    elif args.mode == 'style':
        run_style_reconstruction(args)
    elif args.mode == 'visualize':
        run_feature_visualization(args)
    elif args.mode == 'nst':
        run_neural_style_transfer(args)
    else:
        parser.print_help()


# style reconstruction -[ init - content   ,  ( sw = 1e3 , tvw = 1e-3 ) ]

# nst - [ init - content ,  (sw = 1e5 , c = 1 , tv = 1e-2 ) 