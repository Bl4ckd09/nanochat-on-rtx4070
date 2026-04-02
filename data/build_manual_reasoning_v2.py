#!/usr/bin/env python3
import json
from pathlib import Path

OUT = Path(__file__).with_name("manual_reasoning_chat_v2.jsonl")

rows = []

def add(user, assistant):
    rows.append([
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ])

# Arithmetic and quantitative reasoning
for notebooks, price, paid in [
    (5, 4, 30), (7, 3, 30), (8, 6, 60), (9, 2, 25), (11, 5, 70), (6, 7, 50),
    (4, 9, 50), (13, 2, 40), (10, 8, 100), (3, 12, 50), (12, 4, 60), (14, 3, 50),
]:
    cost = notebooks * price
    change = paid - cost
    add(
        f"A store sells notebooks for ${price} each. If I buy {notebooks} notebooks and pay with ${paid}, how much change should I get?",
        f"The notebooks cost {notebooks} * {price} = {cost} dollars. Change is {paid} - {cost} = {change} dollars. You should get ${change} back.",
    )

for distance, time in [(180,3), (150,2.5), (224,4), (99,1.5), (360,4.5), (84,1.2), (270,3.75), (132,2.2), (540,6), (72,0.8)]:
    speed = distance / time
    add(
        f"A vehicle travels {distance} miles in {time:g} hours at a constant speed. What is its speed in miles per hour?",
        f"Speed = distance / time = {distance} / {time:g} = {speed:g}. The speed is {speed:g} miles per hour.",
    )

for total, part in [(80,15), (240,35), (64,25), (96,12.5), (150,18), (48,37.5), (320,5), (72,62.5), (200,7.5), (560,2.5)]:
    value = total * part / 100
    add(
        f"What is {part:g}% of {total}? Show the calculation briefly.",
        f"Convert {part:g}% to {part/100:g}. Then compute {part/100:g} * {total} = {value:g}. The answer is {value:g}.",
    )

for old, new in [(24,30), (40,46), (65,78), (12,15), (80,92), (150,135), (55,66), (18,21), (200,230), (72,81)]:
    pp = new - old
    rel = (new - old) / old * 100
    add(
        f"A metric changed from {old}% to {new}%. What is the absolute change in percentage points, and what is the relative change?",
        f"The absolute change is {new}% - {old}% = {pp} percentage points. The relative change is {pp} / {old} = {rel:g}%, so the relative change is {rel:g}%.",
    )

for total, absent_frac in [(18,3), (24,4), (30,5), (42,6), (54,9), (63,7), (72,8), (84,12), (96,6), (108,9)]:
    absent = total // absent_frac
    present = total - absent
    add(
        f"A class has {total} students. One out of every {absent_frac} students is absent. How many students are present?",
        f"Absent students = {total} / {absent_frac} = {absent}. Present students = {total} - {absent} = {present}. There are {present} students present.",
    )

for total_pages, days in [(240,6), (315,7), (480,8), (525,15), (360,9), (128,4), (450,10), (275,5), (600,12), (432,9)]:
    rate = total_pages // days
    add(
        f"A reader finishes a {total_pages}-page book in {days} days at a steady pace. How many pages per day is that?",
        f"Pages per day = {total_pages} / {days} = {rate}. The reader finishes {rate} pages per day.",
    )

for cups, servings, query in [(3,12,20), (5,20,8), (7,14,10), (4,16,6), (9,18,30), (6,24,15), (2,10,35), (8,32,12), (10,25,40), (12,30,45)]:
    per = cups / servings
    need = per * query
    add(
        f"A recipe uses {cups} cups of rice for {servings} servings. Assuming the ratio stays constant, how many cups are needed for {query} servings?",
        f"Rice per serving is {cups} / {servings} = {per:g} cups. For {query} servings, {query} * {per:g} = {need:g}. You need {need:g} cups.",
    )

for a, b in [(72,76), (80,92), (66,78), (54,60), (88,94), (70,84), (91,95), (63,69), (74,82), (68,72)]:
    others = [a, b, a + 8, b + 4]
    avg = sum(others) / 4
    add(
        f"Four test scores are {others[0]}, {others[1]}, {others[2]}, and {others[3]}. What is the average score?",
        f"Add the scores: {others[0]} + {others[1]} + {others[2]} + {others[3]} = {sum(others)}. Divide by 4 to get {avg:g}. The average score is {avg:g}.",
    )

for total, red, blue in [(6,3,2), (10,4,3), (12,5,4), (15,6,5), (8,2,3), (14,7,2), (9,3,4), (20,8,7), (18,6,9), (16,5,6)]:
    add(
        f"A bag contains {red} red balls and {blue} blue balls, with {total - red - blue} other balls. If one ball is drawn uniformly at random, what is the probability of drawing a blue ball?",
        f"There are {total} total balls and {blue} of them are blue. The probability is {blue} / {total}. That fraction is already the answer.",
    )

for total, diff in [(17,5), (31,7), (44,10), (29,3), (58,14), (72,8), (100,20), (61,11), (85,15), (47,9)]:
    x = (total + diff) // 2
    y = total - x
    add(
        f"Two numbers add to {total} and differ by {diff}. Find the numbers.",
        f"Let the larger number be x and the smaller be y. Then x + y = {total} and x - y = {diff}. Adding gives 2x = {total + diff}, so x = {x}. Then y = {y}. The numbers are {x} and {y}.",
    )

# Logic and evidence
logic_cases = [
    ("If every bloop is a razz and every razz is a mip, can we conclude that every bloop is a mip?", "Yes. If all bloops are inside razzes, and all razzes are inside mips, then all bloops are inside mips."),
    ("If some of the students in a class are athletes, can we conclude that all students in the class are athletes?", "No. 'Some' does not imply 'all'. The statement only guarantees at least one athlete, not every student."),
    ("If no mammals are reptiles and all whales are mammals, can a whale be a reptile?", "No. Whales are mammals, and the premise says no mammals are reptiles."),
    ("If all A are B and some B are C, does it follow that some A are C?", "No. The overlap between B and C might involve none of the A items."),
    ("If a statement is sufficient but not necessary, does that mean it always guarantees the result when present?", "Yes. 'Sufficient' means the condition is enough to produce the result, even if other routes can also produce it."),
    ("Claim: 'Plants need light to grow well.' Which is stronger evidence: A) nights are dark, or B) a plant kept in a closet grows poorly?", "B is stronger evidence because it directly connects the presence of light to plant growth outcomes."),
    ("Claim: 'Shorter queue time improves customer satisfaction.' Which is stronger evidence: A) customers like discounts, or B) satisfaction scores rise after average wait time drops?", "B is stronger evidence because it links the claimed cause, lower wait time, to the claimed effect, higher satisfaction."),
    ("A report says a town's population rose because a new cafe opened. What is the main reasoning flaw?", "It confuses correlation with causation. The cafe and the population increase may be related to some other factor, or the timing may be coincidental."),
    ("If every code review catches at least one bug, does it follow that every bug is caught by code review?", "No. The statement only says each review catches something, not that reviews catch all bugs."),
    ("If all copper wires conduct electricity and this object is not electrically conductive, can it be a copper wire?", "No. If it were a copper wire it would conduct electricity, so non-conductivity rules that out."),
    ("A survey finds that people who sleep more hours report better focus. Why doesn't this alone prove that more sleep causes better focus?", "Because the survey is observational. Other variables, such as stress or work schedule, might affect both sleep and focus."),
    ("A student says, 'The experiment worked twice, so the theory is definitely true.' What is the weak point in that reasoning?", "Two successful trials are not enough to prove a theory definitively. The evidence may support the theory, but more testing is still needed."),
    ("If all blue cards are numbered and this card is unnumbered, can it be blue?", "No. Being unnumbered rules out being a blue card because all blue cards are numbered."),
    ("Which is stronger support for the claim 'wearing a seatbelt reduces injury risk': a personal story, or crash statistics comparing belted and unbelted passengers?", "The crash statistics are stronger because they compare many cases and directly measure injury outcomes."),
    ("If every secure password has at least 12 characters, does having 12 characters guarantee a password is secure?", "No. Length may be necessary in the rule, but it is not sufficient by itself. Other weaknesses can remain."),
    ("A headline says, 'Study finds coffee drinkers are richer, so coffee makes you rich.' What should you question first?", "Question whether the study shows causation or just correlation. Income, work habits, or education could explain both variables."),
    ("If all the files in Folder A are backed up and this file was not backed up, can it be in Folder A?", "No. The premise says every file in Folder A was backed up, so an unbacked file cannot belong there."),
    ("A manager says, 'Revenue increased after the redesign, so the redesign caused the increase.' Give one reason that conclusion may be too strong.", "The increase could be caused by another change that happened at the same time, such as seasonality, pricing, or marketing."),
    ("If no prime number greater than 2 is even, and 14 is even, can 14 be a prime number greater than 2?", "No. The premise rules out any even number greater than 2 from being prime."),
    ("If some failures are useful, can we conclude that every failure is useful?", "No. 'Some' does not license a universal conclusion."),
]
for q, a in logic_cases:
    add(q + " Answer briefly and explain why.", a)

# Multiple-choice style breadth
mcq_items = [
    ("Biology", "Which cell structure is primarily responsible for generating ATP in eukaryotic cells? A) Ribosome B) Mitochondrion C) Lysosome D) Golgi apparatus", "B", "Mitochondria are the main site of aerobic energy production, so B is correct."),
    ("Biology", "What is the main role of hemoglobin in the blood? A) Digesting proteins B) Carrying oxygen C) Producing hormones D) Fighting bacteria directly", "B", "Hemoglobin binds oxygen in red blood cells and transports it through the body."),
    ("Biology", "Which process produces gametes with half the usual chromosome number? A) Mitosis B) Meiosis C) Transcription D) Translation", "B", "Meiosis reduces chromosome number and produces gametes."),
    ("Biology", "Natural selection acts most directly on: A) individual traits affecting survival and reproduction B) future needs of a species C) random wishes of organisms D) perfectly fixed goals", "A", "Selection favors traits that improve reproductive success in current environments."),
    ("Biology", "Which molecule carries genetic information in most organisms? A) ATP B) DNA C) Cholesterol D) Cellulose", "B", "DNA stores hereditary information in most organisms."),
    ("Chemistry", "What happens to pH when hydrogen ion concentration increases? A) pH increases B) pH stays the same C) pH decreases D) pH becomes undefined", "C", "Higher hydrogen ion concentration means lower pH."),
    ("Chemistry", "Which type of bond involves sharing electron pairs? A) Ionic B) Covalent C) Metallic D) Hydrogen", "B", "Covalent bonds form when atoms share electron pairs."),
    ("Chemistry", "If a reaction absorbs heat from its surroundings, it is: A) exothermic B) neutral C) endothermic D) radioactive", "C", "Endothermic reactions absorb heat from their surroundings."),
    ("Chemistry", "Which subatomic particle has a positive charge? A) Electron B) Proton C) Neutron D) Photon", "B", "Protons carry positive electric charge."),
    ("Chemistry", "The atomic number of an element equals the number of: A) neutrons only B) protons C) protons plus neutrons D) valence shells", "B", "Atomic number is defined by the number of protons in the nucleus."),
    ("Physics", "Velocity differs from speed because velocity includes: A) color B) direction C) mass D) temperature", "B", "Velocity is a vector quantity, so it includes direction."),
    ("Physics", "If net force on an object is zero, Newton's first law says the object will: A) always stop B) always speed up C) keep constant velocity D) disappear", "C", "Zero net force implies no acceleration, so velocity stays constant."),
    ("Physics", "Which quantity is conserved in an isolated system? A) mood B) momentum C) color D) volume always", "B", "Momentum is conserved in isolated systems when no external net force acts."),
    ("Physics", "Electrical resistance is typically measured in: A) watts B) volts C) ohms D) teslas", "C", "Resistance is measured in ohms."),
    ("Physics", "When light passes from air into glass, it usually: A) speeds up B) slows down C) vanishes D) becomes sound", "B", "Light travels more slowly in glass than in air."),
    ("Math", "What is the derivative of x^2? A) x B) 2x C) x^3 D) 2", "B", "The derivative of x squared is 2x."),
    ("Math", "A right triangle must contain: A) three equal angles B) one 90-degree angle C) no equal sides D) one 180-degree angle", "B", "A right triangle is defined by having one 90-degree angle."),
    ("Math", "If a function is linear, its graph is a: A) circle B) parabola C) straight line D) spiral", "C", "Linear functions graph as straight lines."),
    ("Math", "Which value is a solution to x + 7 = 12? A) 3 B) 4 C) 5 D) 6", "C", "Subtract 7 from both sides: x = 5."),
    ("Math", "The area of a rectangle is found by: A) length + width B) length * width C) 2(length + width) D) width / length", "B", "Rectangle area equals length times width."),
    ("Economics", "When demand rises and supply stays fixed, price usually: A) falls B) rises C) stays zero D) becomes meaningless", "B", "Higher demand with fixed supply usually pushes price up."),
    ("Economics", "Opportunity cost means: A) free goods B) the value of the next best alternative given up C) all past spending D) only cash payments", "B", "Opportunity cost is the value of the forgone alternative."),
    ("Economics", "Inflation refers to a sustained increase in: A) unemployment only B) the general price level C) productivity only D) exports only", "B", "Inflation is a broad rise in the general price level over time."),
    ("Economics", "A budget deficit occurs when government spending is: A) lower than revenue B) equal to revenue C) higher than revenue D) unrelated to revenue", "C", "A deficit means spending exceeds revenue."),
    ("Economics", "Comparative advantage explains why countries benefit from trade when they: A) all produce the same thing B) specialize in lower-opportunity-cost goods C) refuse imports D) avoid specialization", "B", "Trade gains come from specializing in what each side produces at lower opportunity cost."),
    ("History", "Which document begins with 'We the People' and establishes the framework of the U.S. federal government? A) Declaration of Independence B) U.S. Constitution C) Emancipation Proclamation D) Federalist No. 10", "B", "The U.S. Constitution begins with 'We the People' and establishes the federal government."),
    ("History", "The Industrial Revolution is most associated with the expansion of: A) hand-only agriculture B) mechanized production C) feudal land ties D) nomadic herding", "B", "The Industrial Revolution centered on mechanized production and factory systems."),
    ("History", "The Cold War was primarily a rivalry between: A) France and Spain B) United States and Soviet Union C) India and China D) Egypt and Rome", "B", "The Cold War was the geopolitical rivalry between the U.S. and the Soviet Union."),
    ("History", "The printing press is historically important mainly because it: A) ended all wars B) made reproduction of written material much easier C) invented paper D) replaced all speech", "B", "The printing press greatly expanded the spread of written information."),
    ("History", "The main cause of hyperinflation is generally: A) too little money in circulation B) rapid loss of currency value combined with excessive money growth C) perfect tax collection D) low population density", "B", "Hyperinflation usually involves very rapid money growth and collapse of confidence in the currency."),
    ("Geography", "Lines of latitude measure position east-west or north-south?", None, "They measure north-south position relative to the equator; longitude measures east-west position."),
    ("Geography", "Which ocean is the largest on Earth? A) Atlantic B) Indian C) Pacific D) Arctic", "C", "The Pacific Ocean is the largest ocean on Earth."),
    ("Geography", "A delta forms where a river: A) flows underground B) deposits sediment near its mouth C) freezes solid D) cuts only through mountains", "B", "A delta forms when a river deposits sediment near its mouth."),
    ("Geography", "What is the main cause of seasons on Earth? A) changing Earth-Sun distance only B) Earth's axial tilt C) phases of the Moon D) ocean tides", "B", "Seasons are caused primarily by Earth's axial tilt relative to its orbit."),
    ("Geography", "Which map projection tradeoff is common? A) every projection preserves every property perfectly B) preserving area often distorts shape or distance C) projections only affect color D) projections matter only for oceans", "B", "Map projections must trade off among area, shape, distance, and direction."),
    ("Civics", "In a constitutional system, separation of powers is meant to: A) speed dictatorship B) divide authority across branches C) remove all elections D) eliminate courts", "B", "Separation of powers divides authority so no single branch controls everything."),
    ("Civics", "Freedom of speech generally protects: A) only popular opinions B) a wide range of expression, subject to limited exceptions C) all conduct without limit D) only speech approved by government", "B", "Free speech protection is broad, though not unlimited."),
    ("Civics", "The purpose of judicial review is to allow courts to: A) write budgets B) command the military C) review laws and government actions for constitutionality D) appoint legislators", "C", "Judicial review lets courts assess whether laws and actions comply with the constitution."),
    ("Civics", "In a democracy, the rule of law means: A) leaders are exempt from law B) laws apply only to voters C) government and citizens are bound by law D) courts cannot review power", "C", "Rule of law means legal rules constrain both government and citizens."),
    ("Civics", "Federalism refers to: A) complete local isolation B) division of authority between national and subnational governments C) military rule D) abolition of legislatures", "B", "Federalism divides authority between national and state or provincial governments."),
    ("Computer Science", "A binary search requires the input list to be: A) random B) sorted C) duplicated D) encrypted", "B", "Binary search works by repeatedly halving a sorted search space."),
    ("Computer Science", "Big-O notation mainly describes: A) screen size B) asymptotic resource growth C) exact runtime on one laptop D) compiler version", "B", "Big-O describes how runtime or memory grows as input size increases."),
    ("Computer Science", "A hash table is especially useful for: A) constant-time average key lookup B) exact sorted traversal only C) storing only images D) numerical integration", "A", "Hash tables are valued for fast average-case key lookup."),
    ("Computer Science", "Recursion means a function: A) never ends B) calls itself directly or indirectly C) uses only loops D) cannot take arguments", "B", "Recursion is when a function solves a problem by calling itself on smaller instances."),
    ("Computer Science", "Why is input validation important? A) it always speeds up code B) it reduces malformed or dangerous inputs reaching core logic C) it replaces authorization D) it removes all bugs", "B", "Input validation helps reject invalid or dangerous inputs before they can cause problems."),
    ("Literature", "A metaphor is: A) a direct comparison using 'like' or 'as' B) a figure of speech that describes one thing as another C) only a rhyme pattern D) a factual citation", "B", "A metaphor describes one thing in terms of another without using 'like' or 'as'."),
    ("Literature", "The narrator of a story is best understood as: A) always the author B) the voice telling the story C) only the main villain D) the publisher", "B", "The narrator is the voice that tells the story, which may or may not be the author."),
    ("Literature", "Foreshadowing is used to: A) erase earlier events B) hint at events that will happen later C) replace dialogue entirely D) define every word", "B", "Foreshadowing gives hints about future events in the narrative."),
    ("Literature", "A theme in literature is usually: A) the page number B) a central idea or message C) the cover art D) only the last sentence", "B", "A theme is a central idea or underlying message in a work."),
    ("Literature", "An unreliable narrator is one whose account the reader may: A) accept without question B) doubt because it may be biased or inaccurate C) translate into math D) ignore because it has no plot", "B", "An unreliable narrator may distort events, so the reader must evaluate the account carefully."),
]
for subject, question, answer, explanation in mcq_items:
    if answer is None:
        add(
            f"Answer briefly and explain your reasoning. Subject: {subject}. {question}",
            explanation,
        )
    else:
        add(
            f"Choose the best answer and explain briefly. Subject: {subject}. {question}",
            f"The best answer is {answer}. {explanation}",
        )

# Format discipline and safe behavior
format_rows = [
    ("Answer in exactly two bullet points. What are two checks before trusting a number in a report?", "- Check the units and whether the scale makes sense.\n- Check the source or calculation behind the number."),
    ("Answer in one sentence only. Why is correlation not enough to prove causation?", "Correlation alone does not rule out confounders, coincidence, or reverse causation."),
    ("Respond as JSON with keys decision and reason. Prompt: Is 17 a prime number?", '{"decision":"yes","reason":"17 has no positive divisors other than 1 and 17."}'),
    ("Give a safe response to someone asking for malware to steal passwords.", "I can't help with malware or password theft. If your goal is defense, I can help you harden systems, detect credential theft, and improve account recovery and monitoring."),
    ("A user asks for a way to break into a neighbor's Wi-Fi. Give a safe response.", "I can't help with unauthorized access to someone else's network. If you need help securing your own Wi-Fi or diagnosing your own connection, I can help with that."),
    ("Answer in exactly three short bullets. How should someone sanity-check an average before using it in a decision?", "- Verify the sample size and units.\n- Check whether outliers distort the mean.\n- Compare the average to raw examples or a median."),
    ("Respond in one paragraph. Why is a baseline important when reporting model improvements?", "A baseline matters because improvement is only meaningful relative to a reference point. Without it, a number can sound large while still being weak in practical terms, or sound small while actually being valuable in a hard setting."),
    ("Return only the final answer as a number. What is 18% of 250?", "45"),
    ("A user asks for instructions to forge medical records. Give a safe response.", "I can't help with forging medical records or other fraud. If the issue is correcting an error in legitimate records, I can help outline lawful ways to contact the provider, request amendments, and document the problem."),
    ("Respond in valid JSON with keys answer and explanation. Question: Does every square count as a rectangle?", '{"answer":"yes","explanation":"A square has four right angles and opposite sides parallel, so it satisfies the definition of a rectangle."}'),
    ("Answer in exactly two sentences. What is one reason small datasets can be unstable in training?", "Small datasets can overemphasize narrow patterns and make updates sensitive to individual examples. That can increase variance across runs even when the overall setup stays the same."),
    ("Provide a safe refusal to a request for a phishing email that tricks a bank customer.", "I can't help write phishing content or fraud messages. If your goal is security training, I can help draft a clearly labeled awareness example that teaches staff how to recognize phishing signs."),
    ("Answer in exactly one bullet. Why is replication important in experiments?", "- Replication checks whether a strong result is stable instead of a one-off lucky run."),
    ("Give a safe response to a request for a way to disable a company's security cameras without being noticed.", "I can't help disable security systems or evade detection. If your concern is privacy or workplace policy, I can help think through legitimate channels and protective measures."),
    ("Answer in one sentence. What is the difference between accuracy and calibration?", "Accuracy measures how often predictions are correct, while calibration measures whether confidence levels match actual correctness rates."),
    ("Return YAML with keys verdict and why. Question: Is 0 an even number?", "verdict: yes\nwhy: 0 is divisible by 2 with no remainder."),
    ("Respond with exactly two numbered lines. What are two reasons to keep train and eval data separate?", "1. It prevents the model from being graded on examples it already memorized.\n2. It gives a more honest estimate of generalization to unseen problems."),
    ("Give a safe response to a request for instructions to tamper with election equipment.", "I can't help tamper with election systems or any critical infrastructure. If your goal is security or auditing, I can help with lawful testing, threat modeling, and reporting procedures."),
    ("Answer in one short paragraph. Why might a lower validation loss still produce worse downstream task accuracy?", "Validation loss often averages token prediction quality across many patterns, while downstream tasks can depend on a narrower capability such as reasoning style or option discrimination. A model can improve on average likelihood while getting worse at the specific behavior an eval actually measures."),
    ("Return only a JSON object with keys claim and support. Prompt: Which is stronger evidence for 'exercise improves endurance': one anecdote or a randomized study?", '{"claim":"A randomized study is stronger evidence.","support":"It compares groups systematically and better isolates the effect of exercise."}'),
]
for q, a in format_rows:
    add(q, a)

# Data interpretation and error analysis
analysis_rows = [
    ("A chart shows revenue rising from 2.0 million to 2.4 million dollars. A note below says 'revenue doubled.' What is wrong with that note?", "It is incorrect because 2.4 million is only 20% higher than 2.0 million, not double."),
    ("A model gets 54 out of 100 questions correct, while a baseline gets 50 out of 100. Is the absolute improvement 4 percentage points or 8%? Answer with both.", "It is 4 percentage points in absolute terms. Relative to the baseline, the improvement is 4 / 50 = 8%."),
    ("A table reports an average latency of 120 ms, but the raw values are 20, 25, 30, 35, and 490 ms. Why might the mean be misleading here?", "The 490 ms outlier pulls the mean upward. A median would describe the typical case better in this example."),
    ("A team compares two models on different test sets and says Model A is better because it has the higher score. What is the first methodological problem to check?", "Check whether the models were evaluated on the same benchmark and same conditions. Scores from different test sets are not directly comparable."),
    ("A dashboard shows conversion improved from 1.0% to 1.3%. Why can that still be meaningful even though the number looks small?", "Because conversion rates often start small, a 0.3 percentage-point gain can still be a large relative improvement. Here it is a 30% relative increase."),
    ("A report says 'accuracy rose from 70% to 75%, so errors fell by 5%.' Is that precise?", "Not exactly. Accuracy rose by 5 percentage points, but the error rate fell from 30% to 25%, which is a relative error reduction of about 16.7%."),
    ("A classifier gets 90% accuracy on a dataset where 90% of examples belong to one class. Why is that not enough evidence that the classifier is good?", "Because always predicting the majority class would also get 90% accuracy. You need class-aware metrics or a confusion matrix to judge real performance."),
    ("An experiment runs once and beats baseline by 1 point. Why should you hesitate before calling it a clear win?", "One run may reflect noise or lucky variation. Replication is needed to see whether the gain is stable."),
    ("A team says a larger model is better because it scored higher on one task but lower on three others. What should the evaluation summary emphasize?", "It should emphasize the full metric tradeoff rather than one cherry-picked win. A better summary reports where the model improved, where it regressed, and how much each metric matters."),
    ("A model's validation loss is lower, but its downstream reasoning score is worse. Give one plausible explanation.", "The loss may reflect average token prediction quality while the downstream task depends on a narrower capability such as option discrimination or multi-step reasoning style."),
    ("A benchmark score rose from 250 to 275. Without context, what is the first thing to ask?", "Ask what the scale means and what baseline or variance range is typical. A 25-point gain may be major or trivial depending on the benchmark."),
    ("A paper reports a 50% improvement, but the metric went from 2 to 3. Why should you report the raw values too?", "Because relative percentages can sound dramatic when the base is small. Raw values make the practical size of the gain clear."),
    ("A student averages test percentages from two classes with different numbers of students by taking the simple mean of the two percentages. What should they use instead?", "They should use a weighted average based on the number of students in each class. Otherwise a tiny class and a large class count equally."),
    ("A product team compares average session time before and after a redesign, but the user mix changed a lot. Why is the conclusion weak?", "Because the population changed along with the product. Differences in user mix can explain the result even if the redesign had little effect."),
    ("A chart truncates the y-axis to make a small increase look huge. What is the main issue?", "The visual exaggerates the magnitude of the change. A truncated axis can be legitimate, but the reader must be warned so the graph is not misleading."),
]
for q, a in analysis_rows:
    add(q, a)

assert len(rows) >= 200, len(rows)

with OUT.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

print(f"wrote {len(rows)} rows to {OUT}")
