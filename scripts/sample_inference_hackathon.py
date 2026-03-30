"""
Hackathon-provided sample inference script (BrowserGym pattern).
This is the reference pattern from the competition. Our inference.py
adapts this same loop for the SQL environment.
"""

import os
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("HF_TOKEN", ""))

SYSTEM_PROMPT = "You are a helpful agent."
FALLBACK_ACTION = "noop"
TEMPERATURE = 0.1
MAX_TOKENS = 2048
MAX_STEPS = 20

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def build_user_prompt(step, observation, history):
    """Build the user prompt from observation and history."""
    parts = [f"Step {step}. Current observation: {observation}"]
    if history:
        parts.append("History:\n" + "\n".join(history[-5:]))
    return "\n\n".join(parts)


def extract_screenshot_uri(observation):
    """Extract screenshot URI from observation if present."""
    if hasattr(observation, "screenshot_uri"):
        return observation.screenshot_uri
    return None


def parse_model_action(response_text):
    """Parse the model's response into an action string."""
    return response_text.strip()


def main():
    """
    Main inference loop — the exact pattern from the hackathon sample.
    This uses BrowserGymAction as the example; our inference.py uses
    SqlAnalystAction instead.
    """
    # NOTE: This is the hackathon reference. It won't run as-is because
    # BrowserGymAction is from the BrowserGym environment, not ours.
    # See inference.py for our actual working implementation.

    # env = BrowserGymEnv(base_url="...")
    # result = env.reset()
    # observation = result.observation
    # history = []
    #
    # try:
    #     for step in range(1, MAX_STEPS + 1):
    #         user_prompt = build_user_prompt(step, observation, history)
    #         user_content = [{"type": "text", "text": user_prompt}]
    #         screenshot_uri = extract_screenshot_uri(observation)
    #         if screenshot_uri:
    #             user_content.append({
    #                 "type": "image_url",
    #                 "image_url": {"url": screenshot_uri},
    #             })
    #
    #         messages = [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {"role": "user", "content": user_content},
    #         ]
    #
    #         try:
    #             completion = client.chat.completions.create(
    #                 model=MODEL_NAME,
    #                 messages=messages,
    #                 temperature=TEMPERATURE,
    #                 max_tokens=MAX_TOKENS,
    #                 stream=False,
    #             )
    #             response_text = completion.choices[0].message.content or ""
    #         except Exception as exc:
    #             failure_msg = f"Model request failed ({exc}). Using fallback action."
    #             print(failure_msg)
    #             response_text = FALLBACK_ACTION
    #
    #         action_str = parse_model_action(response_text)
    #         print(f"Step {step}: model suggested -> {action_str}")
    #
    #         result = env.step(BrowserGymAction(action_str=action_str))
    #         observation = result.observation
    #
    #         reward = result.reward or 0.0
    #         error_flag = " ERROR" if observation.last_action_error else ""
    #         history_line = (
    #             f"Step {step}: {action_str} -> reward {reward:+.2f}{error_flag}"
    #         )
    #         history.append(history_line)
    #         print(
    #             "  Reward: "
    #             f"{reward:+.2f} | Done: {result.done} | Last action error: "
    #             f"{observation.last_action_error}"
    #         )
    #
    #         if result.done:
    #             print("Episode complete.")
    #             break
    #
    #     else:
    #         print(f"Reached max steps ({MAX_STEPS}).")
    #
    # finally:
    #     env.close()

    print("This is the hackathon reference script. See inference.py for our implementation.")


if __name__ == "__main__":
    main()
