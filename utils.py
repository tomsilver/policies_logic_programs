def run_single_episode(env, policy, record_video=False, video_out_path=None, max_num_steps=100):
    if record_video:
        env.start_recording_video(video_out_path=video_out_path)

    obs = env.reset()
    total_reward = 0.
    
    for t in range(max_num_steps):
        action = policy(obs)
        new_obs, reward, done, debug_info = env.step(action)
        total_reward += reward

        obs = new_obs

        if done:
            break

    env.close()

    return total_reward
