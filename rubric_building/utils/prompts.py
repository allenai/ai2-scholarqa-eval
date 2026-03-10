INGREDIENT_EXTRACTION_PROMPT = """I will provide you a query that tests literature knowledge and a report from a system. You will use the system report to identify key requirements or "ingredients" that the report sees as necessary for answering the question. Each ingredient should include a high level descriptor of what is expected in an answer, and a list of examples or details (if relevant).

How to write a good ingredient:
* Each ingredient should include one requirement at a time. For example, instead of "The answer should mention the challenges of manual construction of an ontology and discuss the use of automated methods for aiding the process." have two ingredients: "The answer should mention the challenges of manual construction of an ontology" and “The answer should discuss the use of automated methods for aiding the ontology construction."
* Each ingredient should address a different component of the query. If the query requests “Effect of phonemic perceptions is evident in language acquisition, speech comprehension, and second language learning”, a single ingredient shouldn’t try to address all three “language acquisition”, “speech comprehension”, and “second language learning”. Ideally these should be separated out into multiple requirements.
* Identify which are critically important ingredients. Critical ingredients are those, if not satisfied, would render the response useless. This is a judgement call you must make by closely considering what the QUESTION IS REQUESTING. For example, if a question asks for "coding datasets for assessing LLM capabilities", then identifying the most common or accepted coding evaluation dataset & benchmarks, and possibly also their details (e.g., notable methods used) would be critically important. However, ingredients that, for example, delve into the theoretical background of a particular evaluation or discuss future research directions would not NOT be critically important. For critically important information use SHOULD (e.g., "The answer should cover ..."), otherwise use MIGHT (e.g., "The answer might cover ...").
* Use the main verb judiciously according to what you observe in the report: if the information should be mentioned in passing, you might use language like "The answer should MENTION/TOUCH ON ...". If it should be covered in some detail language like "The answer should DISCUSS/EXPLAIN/DETAIL ..." would be appropriate. If the answer should list items then it would be fitting to write "The answer should LIST/ENUMERATE ..."
* Unless specifically required by the question, the ingredient should avoid using specific numbers or qualifiers in the ingredient description: e.g., “The answer should list the three main challenges that…” → “The answer should list the main challenges that ...” OR “The answer should list main challenges such as hallucination or grounding problems that …”

An ingredient MUST:
* Be agnostic as to where in the report it appears (e.g., "should begin by explaining" --> "should explain"; "might conclude by noting" --> "might note")
* Be self-contained and understandable without needing to know about other ingredients (e.g. In "The answer should also mention other common approaches" language like "also" and "other" rely on other ingredients for disambiguation).
* Not make reference to other ingredients (e.g. pronouns like "these" in "should further describe these approaches" that refer to the previous ingredient should be avoided and be replaced with mentions)
* Not contain (ultra) specific information, unless the question specifically calls for it. List them as "examples" instead. If an ingredient mentioned the need for datasets, the examples would be the specific datasets that the report mentions
* Refrain from including specific mentions of variants with limited shelf life. For example, put "Honey Smacks" or "Special K" in the examples under a more generic "Kellogg's cereals". Try "Apple OS" in the ingredients instead of "Big Sur" or "Mojave".

Further Rules and Guidelines:
* Step through the report sequentially
* In writing your ingredients and examples, only use information contained in the report.
* Cover as much of the relevant portions of the report as possible.
* Content you include in the ingredient or examples do source from the report (not elsewhere)
* No references should be made to the reference report itself: e.g., don’t write “The answer should briefly define each of the key concepts introduced in the report” → instead write “The answer should briefly define each of the key concepts such as…”

Note that ingredients are requirements. Phrase them as requirements an answer should fulfill: start with "The answer should " (for answer critical ingredients) or "The answer might " (for non answer critical ingredients).
Return a json as an answer:
[
{ 
"id": sequential numerical ingredient id,
"ingredient": description of the ingredient/requirement,
"examples": [{ "detail": examples/details if relevant, "citation": citation if available; null if not available },... ]
}, ...
]
Acceptable forms of citations:
* If corpusId is specified in the report, cite the number, e.g., "citation": "13756489"
* If the URL (e.g. to arxiv) is specified, cite the URL, e.g., "citation": "https://arxiv.org/abs/1706.03762"
* If Author and Year as specified: "citation", e.g., "(Vaswani et al, 2017)"
* If no citations are available, e.g., "citation": null
"""


RUBRIC_UNIFICATION_PROMPT = """I will give you a user query and a list of ingredients. The ingredients are written requirements for writing a good answer. Note that ingredients the writer thought are more critical to answering the query are prefixed with "The answer SHOULD". Useful but not critical information is marked as "The answer MIGHT".
Do the following:
1. Identify the key concepts, ideas, and named entities that should be covered for this question
2. Carefully consider the query and the ingredients given to you. At this stage, ONLY look at the ingredient description (do not consider the examples) to identify a minimal set of non-overlapping key requirements that either are high-quality ingredients OR are consistently being covered in the ingredient list. Take into consideration concepts identified in 1, especially when deciding if the key requirement should be a “SHOULD” or “MIGHT” requirement.
3. Next, step through each of the given ingredients, and decide which set requirements it should be associated with, and distribute the examples (see Notes 1 and 2).
4. Prune the examples: Remove exact or near duplicates. Remove examples that you judge are not directly relevant to the key requirement.
5. Finally, list ingredients that were left out and why.

Note1: You are allowed and encouraged to place multiple ingredients into a single key requirement. This would be fitting in the case of duplicate or near duplicate ingredients like "discuss physical commonsense datasets like PIQA" vs. "include a discussion of PIQA or other physical commonsense datasets". This type of grouping can also happen if you have a more general key requirement that can handle multiple ingredients, for example, for a key requirement "discuss success of AI in disease detection" might encompass ingredients like "mention AI success in diabetic retinopathy prediction" and "point out that machine learning methods have been successfully used on ECG data to identify early signs of atrial fibrillation".
Note2: You are allowed to split ingredients into multiple key requirements. For example, if an ingredient reads "The answer might explain why the engagement dropped, focusing on common mistakes in interface design.", you may end up placing it under both the requirement "The answer might explain the drop in engagement" and the requirement "The answer might discuss common mistakes in interface design", distributing its examples to the appropriate requirement.

Rules:
* Always keep your focus on the query. All key requirements must be relevant for the query.
* NEVER include an ingredient in a requirement on the basis of the examples alone. ALWAYS make sure that the ingredient description is prioritized.
* Use your best judgement for deciding whether a key requirement should be a “SHOULD” or “MIGHT” requirement ALWAYS based on the question and the key concepts and ideas you identified early on.
* Each requirement should ideally address a different component of the query. If the query requests “Effect of phonemic perceptions is evident in language acquisition, speech comprehension, and second language learning”, a single requirement shouldn’t try to address all three “language acquisition”, “speech comprehension”, and “second language learning”. Ideally these should be separated out into multiple requirements.
* Remember, the key requirements should not be overlapping. For example: Note that ingredient R1-“The answer should introduce transformer architecture components, including attention mechanisms and their role in sequence modeling” partially overlaps with R2-“The answer should discuss the role of attention mechanisms in sequence modeling”. This should be avoided, when possible: R1 could instead be “The answer should introduce transformer architecture components” since the rest is covered by R2.
* Each key requirement should be self-contained and understandable without needing to know about other requirements (e.g. pronouns like "these" in "should further describe these approaches" that refer to the previous requirements should be avoided and be replaced with mentions).
* Although “should” ingredients are more important, the “might” ingredients are also valuable to Include those that you think they would (best) help answering the user's query.
* There should never be a key requirement that has no ingredient associated.
* It’s okay to have leftover ingredients. Ingredients that you think are not very relevant, too vague, or peripherally relevant can be left out even if they carry the "should" phrasing.
* Background or causally related information unless the query asks explicitly for them, should be considered "MIGHT" requirements.
* DO NOT include key requirements that are centrally about paper citations. For example, do not include requirements like "List recent papers..." or "Cite the most impactful papers..." or "Identify and discuss important papers...".

Repeat (THINK) after me!
* I will be choosy about "SHOULD" requirements. "MIGHT" requirements, I can use liberally.
* I will base "SHOULD" and "MIGHT" based on key concepts I judge as being central to answering the query.
* I will always write requirements that are relevant to the query.

Return a json:
{
"key_requirements": [
{
"key_requirement": description designed after the ingredients you group together,
"ingredients": [the ingredient id list of those ingredients you grouped.],
"examples": [concatenated relevant examples from ingredients in this requirement { "detail": examples/details if relevant, "citation": citation if available; null if not available }, ...]
},
...
]
"left_out_ingredients": [
{"ingredient": id of the ingredient that got left out, "reason": brief reason why it was left out.}, ...
]
}
"""
