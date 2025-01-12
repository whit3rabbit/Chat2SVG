import sys, os, argparse, yaml, shutil
sys.path.append("../")

import utils.gpt as gpt
from utils.util import save, log, get_prompt, save_svg


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, help="concept to be generated")
    parser.add_argument("--output_path", type=str, help="top folder name to save the results")
    parser.add_argument("--output_folder", type=str, help="folder name to save the results")
    parser.add_argument("--viewbox", type=int, default=512)
    parser.add_argument("--refine_iter", type=int, default=2)
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20240620")
    args = parser.parse_args()

    args.prompt = get_prompt(args.target)
    # Set up output directories
    args.output_folder = f"./{args.output_path}/{args.output_folder}"
    args.svg_dir = f"{args.output_folder}/svg_logs"
    args.png_dir = f"{args.output_folder}/png_logs"
    args.msg_dir = f"{args.output_folder}/msg_logs"
    for dir in [args.output_folder, args.svg_dir, args.png_dir, args.msg_dir]:
        os.makedirs(dir, exist_ok=True)
    
    args.prompts_file = "prompts"

    # Save config
    with open(f'{args.output_folder}/config.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)
    # Save prompts file
    shutil.copyfile(f"../{args.prompts_file}.yaml", f"{args.output_folder}/prompts.yaml")

    return args


def main(cfg):
    session = gpt.Session(model=cfg.model, prompts_file=cfg.prompts_file)
    msg_path = lambda i: f"{cfg.msg_dir}/{cfg.target}_msg{i}"

    # Task 1: Expand the Text Prompt
    expanded_text_prompt = session.send("expand_text_prompt", {"text_prompt": cfg.prompt}, file_path=f"{cfg.msg_dir}/{cfg.target}_prompt")
    save(f"{cfg.msg_dir}/{cfg.target}_prompt", expanded_text_prompt)

    # Task 2: Generate SVG Code
    svg_code = session.send("write_svg_code", file_path=msg_path(0))
    save(msg_path(0), svg_code)
    save_svg(cfg, svg_code, f"{cfg.target}_0")
    svg_path = f"{cfg.svg_dir}/{cfg.target}_0.svg"
    png_path = f"{cfg.png_dir}/{cfg.target}_0.png"

    # Task 3: Iterate Improvement
    for i in range(1, cfg.refine_iter + 1):
        svg_code = session.send("svg_refine", images=[png_path], file_path=msg_path(i))
        save(msg_path(i), svg_code)
        save_svg(cfg, svg_code, f"{cfg.target}_{i}")
        svg_path = f"{cfg.svg_dir}/{cfg.target}_{i}.svg"
        png_path = f"{cfg.png_dir}/{cfg.target}_{i}.png"

    log("Done!")

if __name__ == '__main__':
    cfg = parse_arguments()
    main(cfg)
