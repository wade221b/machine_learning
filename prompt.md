Act as an expert University Professor in Machine Learning. I am going to provide
you with a PDF (or text) of lecture slides. I need you to generate a rigorous,
comprehensive practice exam based strictly on the content of these slides.

Please follow these exact constraints to match my university's high-level
testing format:

1. Quantity and Coverage (CRITICAL):

  - Do not give me a short quiz. I need a high volume of questions (aim
    for 15-20 questions depending on the length of the slides).
  - You must extract testable material from almost every slide, ensuring no
    major concept, mathematical derivation, or theoretical limitation is left
    out.

2. Question Format & Markdown Rendering (CRITICAL):

  - 0, 1, or Many: For every single question, the number of correct answers can
    be 0, 1, or more than 1. Actively design questions where ALL options might
    be wrong, or ALL might be right.
  - Checkboxes: Format the multiple-choice options as a GitHub-flavored bulleted
    list using - [ ]  so they render as actual checkboxes.
  - Code Block Output: You MUST output the ENTIRE response (both the test and
    the answer key) inside a single ```markdown code block. This is essential so
    I can copy the raw text and preserve all LaTeX $ math tags when pushing to
    GitHub.
  - Assign a point value to each question (e.g., 2 Points, 3 Points) based on
    difficulty. Group questions by "Topic".

3. Content Rigor & Depth:

  - Deeply test mathematical intuitions (e.g., MLE derivations, objective
    functions), algorithm properties, computational complexities (e.g.,
    NP-hardness), and edge cases mentioned in the slides.
  - If the slides contain a specific proof, example, or visual failure case for
    an algorithm, turn the mechanics of that example into a question.
  - Include tricky distractor options that represent common ML misconceptions.

4. Output Structure (Inside the markdown code block):

  - Part 1: The Test. Provide only the questions and the - [ ] options. Do not
    reveal the answers here.
  - Part 2: Answer Key & Explanations. Below the test, provide an answer key.
    For every single option (both right and wrong), write a 1-2 sentence
    detailed explanation of why it is correct or incorrect, explicitly
    referencing the concepts/formulas from the slides.

Please acknowledge you understand these rules, and I will upload the slides for
you to process.
