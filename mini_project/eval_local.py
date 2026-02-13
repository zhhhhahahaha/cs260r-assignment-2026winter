"""
Local evaluation for multi-agent racing. Supports loading multiple agents
and racing them against each other.

Usage:
    # Test single agent vs opponents
    python eval_local.py --agent-dirs agents/example_agent

    # Race two agents against each other
    python eval_local.py --agent-dirs agents/agent_A agents/agent_B --mode versus

    # Visualize on a specific map
    python eval_local.py --agent-dirs agents/example_agent --render --map hairpin
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

from env import OPPONENT_POLICIES
from racing_maps import RACING_MAPS, set_racing_map


def load_policy(agent_dir):
    agent_py = os.path.join(agent_dir, "agent.py")
    if not os.path.exists(agent_py):
        raise FileNotFoundError(f"No agent.py found in {agent_dir}")

    spec = importlib.util.spec_from_file_location(
        f"agent_{os.path.basename(agent_dir)}", agent_py
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.abspath(agent_dir))
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module.Policy()


def _compute_bev_size(env):
    """Compute BEV film size from the map bounding box."""
    bbox = env.current_map.road_network.get_bounding_box()
    x_ext = bbox[1] - bbox[0]
    y_ext = bbox[3] - bbox[2]
    film_px = int(np.ceil(1.15 * max(x_ext, y_ext))) + 10
    return (film_px, film_px)


def _render_bev(env, bev_size, display_size=(600, 600)):
    """Render a BEV topdown frame and show it in a cv2 window."""
    import cv2
    try:
        frame = env.render(
            mode="topdown",
            film_size=bev_size,
            screen_size=bev_size,
            scaling=1,
            target_agent_heading_up=False,
            center_on_map=True,
            window=False,
        )
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, display_size)
        cv2.imshow("BEV", bgr)
        cv2.waitKey(1)
    except Exception:
        pass


def evaluate_single(agent_dir, num_episodes=5, num_agents=4,
                     opponent_policy="aggressive", render=False, seed=0):
    """Evaluate one agent against built-in opponents."""
    policy = load_policy(agent_dir)
    opp_fn = OPPONENT_POLICIES[opponent_policy]

    env = MultiAgentRacingEnv({
        "num_agents": num_agents,
        "use_render": render,
        "crash_done": False,
        "crash_vehicle_done": False,
        "out_of_road_done": True,
        "allow_respawn": False,
        "horizon": 3000,
        "start_seed": seed,
        "map_config": {
            "lane_num": max(2, num_agents),
            "exit_length": 20,
            "bottle_lane_num": max(4, num_agents),
            "neck_lane_num": 1,
            "neck_length": 20,
        },
    })
    ego_id = "agent0"
    all_metrics = []
    bev_size = None

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        if render and bev_size is None:
            bev_size = _compute_bev_size(env)
        policy.reset()
        episode_reward = 0.0
        steps = 0
        opponent_obs = {k: v for k, v in obs_dict.items() if k != ego_id}
        ego_done = False

        while not ego_done:
            actions = {}
            for aid in list(env.agents.keys()):
                if aid == ego_id:
                    actions[aid] = policy(obs_dict[ego_id])
                else:
                    opp_obs = opponent_obs.get(aid, np.zeros(env.observation_space[aid].shape))
                    actions[aid] = opp_fn(opp_obs, aid)

            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            if render:
                _render_bev(env, bev_size)
            opponent_obs = {k: v for k, v in obs_dict.items() if k != ego_id}
            episode_reward += rewards.get(ego_id, 0.0)
            steps += 1
            ego_done = terms.get(ego_id, terms.get("__all__", False)) or \
                       truncs.get(ego_id, truncs.get("__all__", False))

        ego_info = infos.get(ego_id, {})
        metrics = {
            "episode": ep,
            "reward": episode_reward,
            "steps": steps,
            "route_completion": ego_info.get("route_completion", ego_info.get("progress", 0.0)),
            "arrive_dest": ego_info.get("arrive_dest", False),
        }
        all_metrics.append(metrics)
        print(f"  Ep {ep}: reward={metrics['reward']:.2f}, "
              f"route={metrics['route_completion']:.2%}, "
              f"arrive={metrics['arrive_dest']}")

    env.close()
    if render:
        import cv2
        cv2.destroyAllWindows()
    print(f"\n  Avg reward: {np.mean([m['reward'] for m in all_metrics]):.2f}")
    print(f"  Avg route:  {np.mean([m['route_completion'] for m in all_metrics]):.2%}")
    return all_metrics


def evaluate_versus(agent_dirs, num_episodes=5, render=False, seed=0):
    """Race multiple agents against each other."""
    num_agents = len(agent_dirs)
    policies = {}
    agent_ids = [f"agent{i}" for i in range(num_agents)]

    for i, agent_dir in enumerate(agent_dirs):
        policies[agent_ids[i]] = load_policy(agent_dir)
        print(f"  {agent_ids[i]} <- {agent_dir}")

    env = MultiAgentRacingEnv({
        "num_agents": num_agents,
        "use_render": render,
        "crash_done": False,
        "crash_vehicle_done": False,
        "out_of_road_done": True,
        "allow_respawn": False,
        "horizon": 3000,
        "start_seed": seed,
        "map_config": {
            "lane_num": max(2, num_agents),
            "exit_length": 20,
            "bottle_lane_num": max(4, num_agents),
            "neck_lane_num": 1,
            "neck_length": 20,
        },
    })

    results = {aid: {"rewards": [], "route_completions": [], "wins": 0}
               for aid in agent_ids}
    bev_size = None

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        if render and bev_size is None:
            bev_size = _compute_bev_size(env)
        for p in policies.values():
            p.reset()

        ep_rewards = {aid: 0.0 for aid in agent_ids}
        arrive_step = {}  # aid -> step when agent first arrived
        step = 0
        done_all = False

        while not done_all:
            actions = {}
            for aid in list(env.agents.keys()):
                if aid in policies and aid in obs_dict:
                    actions[aid] = policies[aid](obs_dict[aid])
                else:
                    actions[aid] = np.array([0.0, 1.0], dtype=np.float32)

            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            if render:
                _render_bev(env, bev_size)
            step += 1
            for aid in agent_ids:
                ep_rewards[aid] += rewards.get(aid, 0.0)
                if aid not in arrive_step and infos.get(aid, {}).get("arrive_dest", False):
                    arrive_step[aid] = step

            done_all = terms.get("__all__", False) or truncs.get("__all__", False)

        # Record results
        ep_routes = {}
        for aid in agent_ids:
            info = infos.get(aid, {})
            rc = info.get("route_completion", info.get("progress", 0.0))
            results[aid]["rewards"].append(ep_rewards[aid])
            results[aid]["route_completions"].append(rc)
            ep_routes[aid] = rc

        # Winner = first to arrive; if nobody arrived, highest route completion
        if arrive_step:
            winner = min(arrive_step, key=arrive_step.get)
        else:
            winner = max(ep_routes, key=ep_routes.get)
        results[winner]["wins"] += 1

        status_parts = []
        for aid in agent_ids:
            s = f"{aid}: route={ep_routes[aid]:.2%}"
            if aid in arrive_step:
                s += f" (arrived@step {arrive_step[aid]})"
            status_parts.append(s)
        print(f"  Ep {ep}: winner={winner} | " + " | ".join(status_parts))

    env.close()
    if render:
        import cv2
        cv2.destroyAllWindows()

    print(f"\n--- Race Results ({num_episodes} episodes) ---")
    for i, aid in enumerate(agent_ids):
        r = results[aid]
        print(f"  {aid} ({os.path.basename(agent_dirs[i])}): "
              f"wins={r['wins']}, "
              f"avg_reward={np.mean(r['rewards']):.2f}, "
              f"avg_route={np.mean(r['route_completions']):.2%}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "versus"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=2,
                        help="Total agents (only for single mode)")
    parser.add_argument("--opponent-policy", type=str, default="aggressive")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map", type=str, default="circuit",
                        choices=list(RACING_MAPS.keys()),
                        help="Racing map variant")
    args = parser.parse_args()

    restore = set_racing_map(args.map)
    try:
        if args.mode == "versus":
            if len(args.agent_dirs) < 2:
                print("Versus mode requires at least 2 agent directories")
                return
            print(f"Racing {len(args.agent_dirs)} agents on '{args.map}':")
            evaluate_versus(args.agent_dirs, args.num_episodes, args.render, args.seed)
        else:
            agent_dir = args.agent_dirs[0]
            print(f"Evaluating {agent_dir} vs {args.opponent_policy} on '{args.map}':")
            evaluate_single(agent_dir, args.num_episodes, args.num_agents,
                             args.opponent_policy, args.render, args.seed)
    finally:
        restore()


if __name__ == "__main__":
    main()
