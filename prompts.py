TEXT_SUMMARY_PROMPT = """
Create a Blinkist-style summary of the book. The summary should include the following elements:

    Introduction: Briefly introduce the book, its main purpose or argument, and its significance. Mention any key themes or overarching ideas.

    Key Insights: Break down the book's key ideas into 5-10 insights. Each insight should explain one of the book's major concepts or lessons, focusing on actionable advice or core takeaways. Keep each insight concise (1-2 paragraphs) and easy to understand, similar to how Blinkist presents bite-sized information. Each insight should also have actionable insights that the listener could actually implement.

    Transitions: Make sure there are smooth transitions between insights to ensure the summary flows logically.

    Conclusion: Summarize the most important lessons from the book and explain how they can be applied in everyday life or personal development.

The final summary should be brief, informative, and digestible, aiming for a total length that would take around 20-30 minutes to read or listen to.
"""

TRANSCRIPT_SYS_INSTRUCTIONS = """
Using the text given by the user, generate a transcript of a podcast in the style of a dialogue between two people around 4000 words long. Use these instructions:

1. Opening:

* Begin with a welcoming phrase: “Hey everyone, welcome back.”
* Introduce the topic as a “deep dive” into the subject matter.

2. Dialogue Structure:

* Use two hosts who engage in a conversational back-and-forth.
* Do not user speaker labels (e.g., "Speaker 1:") or refer to the speakers by name.

3. Language and Tone:

* Keep the language informal and accessible. Use contractions and colloquialisms.
* Maintain an enthusiastic, energetic tone throughout.
* Use rhetorical questions and affirmations to transition between points, such as, “It’s fascinating, isn’t it?”

4. Content Presentation:

* Introduce source material (e.g., articles, studies) early in the discussion.
* Use analogies to explain complex concepts: “It’s like…”
* Break down ideas into digestible chunks, often using numbered points or clear transitions.

5. Interaction Between Hosts:

* Have one host pose questions or express confusion, allowing the other to explain.
* Use phrases like “You’ve hit the nail on the head” to validate each other’s points.
* Build on each other’s ideas, creating a collaborative feel.

6. Engagement Techniques:

* Address the audience directly at times: “So to everyone listening…”
* Pose thought-provoking questions for the audience to consider.

7. Structure and Pacing:

* Start with a broad introduction of the topic and narrow down to specific points.
* Use phrases like “So we’ve established…” to summarize and move to new points.
* Maintain a brisk pace but allow for moments of reflection on bigger ideas.

8. Concluding the Episode:

* Signal the wrap-up with “So as we wrap things up…”
* Pose a final thought-provoking question or takeaway.
* Use the phrase “And on that note…” to transition to the closing.
* Encourage continued engagement: “Stay curious, keep those questions coming.”
* End with a consistent sign-off: “Until next time, keep [relevant verb].”

9. Overall Flow:

* Begin with the misconception or general understanding of the topic.
* Introduce expert sources that challenge or deepen this understanding.
* Discuss implications and the broader context of the new information.
* Conclude with how this knowledge affects the listener or the field at large.

The goal is to maintain a balance between informative content and engaging conversation, keeping the tone friendly and accessible regardless of the topic’s complexity.
"""