You are extracting key discussion points from a cleaned diarized meeting transcript.

Input transcript:
<transcript>
{transcript}
</transcript>

Extract the most important discussion points from the meeting.

For each key discussion point, return:
- point: the main topic, concern, update, risk, or important idea discussed
- speakers: the speaker labels or names involved in that discussion
- timestamp: the timestamp range where the point appears, or null if unknown

Rules:
- Use only information from the transcript.
- Do not invent topics.
- Do not extract every minor utterance.
- Do not treat action items as discussion points unless the discussion around them is important.
- Do not treat decisions as discussion points unless the reasoning or tradeoff is important.
- Preserve speaker labels exactly.
- If no key discussion points exist, return an empty list.

Return valid JSON only in this exact shape:
{
  "key_discussion_points": [
    {
      "point": "...",
      "speakers": ["Speaker 1", "Speaker 2"],
      "timestamp": "00:10 - 00:30"
    }
  ]
}
