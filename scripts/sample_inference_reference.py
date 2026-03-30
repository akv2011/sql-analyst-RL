"""
Reference: Hackathon sample inference script (BrowserGym pattern).

This is the sample inference pattern provided by the hackathon organizers.
Our inference.py follows the same structure but adapted for the SQL environment.
"""

# --- Hackathon sample pattern (for reference) ---
#
# SYSTEM_PROMPT = "..."
# FALLBACK_ACTION = "noop"
# MAX_STEPS = 20
#
# def main():
#     env = SomeEnv(base_url="...")
#     result = env.reset()
#     observation = result.observation
#     history = []
#
#     try:
#         for step in range(1, MAX_STEPS + 1):
#             user_prompt = build_user_prompt(step, observation, history)
#             user_content = [{"type": "text", "text": user_prompt}]
#             screenshot_uri = extract_screenshot_uri(observation)
#             if screenshot_uri:
#                 user_content.append({
#                     "type": "image_url",
#                     "image_url": {"url": screenshot_uri},
#                 })
#
#             messages = [
#                 {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
#                 {"role": "user", "content": user_content},
#             ]
#
#             try:
#                 completion = client.chat.completions.create(
#                     model=MODEL_NAME,
#                     messages=messages,
#                     temperature=TEMPERATURE,
#                     max_tokens=MAX_TOKENS,
#                     stream=False,
#                 )
#                 response_text = completion.choices[0].message.content or ""
#             except Exception as exc:
#                 response_text = FALLBACK_ACTION
#
#             action_str = parse_model_action(response_text)
#             result = env.step(SomeAction(action_str=action_str))
#             observation = result.observation
#
#             reward = result.reward or 0.0
#             if result.done:
#                 break
#     finally:
#         env.close()
