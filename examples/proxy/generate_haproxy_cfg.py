import argparse
import os

from jinja2 import Environment, FileSystemLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HAProxy config from Jinja template."
    )
    parser.add_argument(
        "--output", "-o",
        default="/etc/haproxy/haproxy.cfg",
        help="Output file path (e.g., /etc/haproxy/haproxy.cfg)",
    )
    parser.add_argument(
        "--nodes", "-n",
        required=True,
        help="newline-separated list of node IPs, e.g. '10.0.0.1\\n10.0.0.2'",
    )
    parser.add_argument(
        "--proxy-port", "-p",
        type=int,
        required=True,
        help="Proxy bind port (e.g. 8000)",
    )
    parser.add_argument(
        "--vllm-port", "-v",
        type=int,
        required=True,
        help="vLLM server port (e.g. 8080)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse node IPs
    node_ips = args.nodes.splitlines()
    if not node_ips:
        raise ValueError("No valid node IPs provided via --nodes")

    if args.proxy_port == args.vllm_port:
        raise ValueError("Proxy port and vLLM port must be different")

    print(f"Detected nodes: {node_ips}")

    # Load Jinja2 template (same directory as script)
    template_dir = os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("haproxy.cfg.j2")

    # Render
    rendered_cfg = template.render(
        proxy_port=args.proxy_port,
        vllm_port=args.vllm_port,
        vllm_servers=node_ips,
    )

    # Write output
    with open(args.output, "w") as f:
        f.write(rendered_cfg)

    print(f"HAProxy config written to: {args.output}")
    print(f"Config content:\n{rendered_cfg}")


if __name__ == "__main__":
    main()
