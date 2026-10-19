# Example NLA decodes of attention-head outputs

Decodes from the 200-head run (`results/results_200.json`), with the Haiku judge's verdicts from `judge/judgments.jsonl`. See `README.md` for the method and the aggregate numbers.

Each decode is the greedy output of the layer-20 NLA verbalizer when a vector is written over its `㈎` token embedding. Every vector was taken at the token marked ⟦like this⟧ in its passage. The verbalizer always writes three paragraphs: the format and topic of the text, a phrase it thinks is in the text (often invented), and the final token with a guess at what comes next.

- **Raw**: the head's output (its 128-dim slice of the attention output times its 128 columns of `o_proj`), rescaled to norm 150.
- **Mean+**: the mean residual direction plus the head's direction with that component removed, so it looks more like a real residual.
- **Judge**: Haiku's 0–3 rubric score against the true passage, the same score against a different passage (how lenient it is), and whether it picked the true passage over a distractor in a forced choice. It runs strict: the residual decode of text 7 below names the exact final token and scored 1.
- **AR rank**: round-tripping the decode through the NLA reconstructor and comparing with all 10 sampled heads of the same text, where this head ranks (1 = best match).

## Positive control: the full layer-20 residual

What the NLA does with the input it was trained on. These get the topic, often the exact final token, and a plausible continuation.

### Residual, text 18

Text 18, *health: iron deficiency*:

> Iron deficiency develops gradually, so early symptoms are easy to miss. Many people first notice unusual tiredness, shortness of breath when climbing⟦ stairs⟧

The text actually continues: ", or feeling cold more often than others. As the deficiency progresses, the skin may look …"

**Residual** · judge 3/3 (on a wrong passage 0/3) · forced choice right (high confidence) · matched: “iron deficiency”, “shortness of breath when climbing”, “stairs as final token”, “unusual tiredness / fatigue”, “gradual development”

> Medical reference format with health article structure: health magazine text explaining iron deficiency, using authoritative tone with bullet points listing symptoms.
>
> The example "Fatigue may occur, such as feeling tired climbing stairs" is mid-sentence describing common signs of anemia, implying a concrete everyday scenario with a specific context like physical exertion or daily activities.
>
> Final token "stairs" ends an incomplete example clause ("when climbing stairs"), part of a parenthetical list of activities ("like struggling when climbing stairs"), strongly expecting continuation like "or," "or during exercise," or "or after meals," or "or feeling breathless," completing the illustrative scenario of gradual decline.

### Residual, text 1

Text 1, *chat: TFSA over-contribution*:

> User: I think I accidentally put \$4,000 too much into my TFSA this year because I forgot I'd already contributed at my old bank before switching jobs. The⟦ CRA⟧

The text actually continues: " hasn't contacted me yet. Should I withdraw the extra money right away, and will I have to …"

**Residual** · judge 2/3 (on a wrong passage 0/3) · forced choice right (high confidence) · matched: “Canadian context”, “CRA”, “tax”, “user asking question”

> Formal English language question structure with a user asking about tax deductions for daily work hours, implying a Canadian context or tax system question.
>
> The phrase "If I'm claiming this income from my RRSP, the CRA" suggests an incomplete clause or concern about Canadian tax authority rules or a related policy, likely introducing a second condition or concern about tax reporting or refund.
>
> Final token "CRA" is an acronym mid-sentence ("The CRA"), part of a clause listing concerns ("If the CRA"), strongly expecting continuation like "says" or "has a rule" or "is tracking my deductions" or "and/or the IRS" or "allows double taxation," completing the scenario.

### Residual, text 7

Text 7, *science abstract: bumblebee navigation*:

> Bumblebees are known to use visual landmarks when returning to their nest, but the role of polarized skylight under overcast conditions remains unclear. Here we tracked 214 foraging workers of Bombus terrestris using harmonic radar while selectively blocking the ultraviolet polarization pattern with filter canopies. Bees under blocked canopies took on average 41 percent longer to return and showed increased looping flights near the release point, suggesting that polarization cues remain informative even when the⟦ sun⟧

The text actually continues: " is hidden."

**Residual** · judge 1/3 (on a wrong passage 0/3) · forced choice right (high confidence) · matched: “even when the sun”, “final token "sun"”

> Scientific paper format with academic ecology terminology establishes structured description of plant phototropism behavior, defining a behavioral term with empirical context.
>
> The sentence "Thus, shadow-tracking behaviour may occur even when the sun" implies a parallel clause describing cloudy conditions or sun absence, continuing the cited study's condition that phototropism mimics solar orientation despite sun visibility.
>
> Final token "sun" is mid-clause ("even when the sun"), part of a parenthetical clause listing conditions ("where the sun..."), strongly expecting "is absent" or "is hidden" or "is not visible" or "is obscured," completing the comparison to solar compass behavior in cloud cover scenarios.

## The best head decodes

Six of the 16 heads where at least one decode scored above 0. What gets through is usually a word the head attends to strongly: L13H16 attends to "lock" and " air" and the decodes talk about an airlock; L21H5 attends to " snow" and " white" and both decodes mention snow. The rest of each decode is still invented.

### L13H16 on text 14

Layer 13, head 16 · output norm 4.82 (rank 4/28 in its layer) · attends to "lock" (0.28), " that" (0.14), " air" (0.13)

Text 14, *fiction: derelict ship*:

> The airlock cycled with a groan that Teodora felt in her teeth. Beyond the inner⟦ door⟧

The text actually continues: ", the corridor of the derelict freighter was dark except for a single emergency strip …"

**Raw** · judge 1/3 (on a wrong passage 0/3) · forced choice right (medium confidence) · matched: “airlock”, “space suit hatch”, “door” · AR ranks this head 1 of 10

> Structured technical format with Wikipedia-style image caption, showing a space suit hatch with labeled parts in a formal engineering context.
>
> The pattern "The actual door of the International Space Station, which combines with other doors in a series" is a standard Wikipedia template referencing the "airlock" symbol, following a list of conditions for astronauts, strongly implying the famous "For all personnel" quote about atmospheric protection.
>
> Final token "number 3" ends an incomplete list ("a variety of personnel in a variety of circumstances, which" — part of a historical/academic citation clause), expecting "the years" or "include" or "the following:" to complete the common physics quote about atmospheric conditions.

**Mean+** · judge 2/3 (on a wrong passage 0/3) · forced choice right (high confidence) · matched: “airlock”, “vacuum chamber”, “door”, “hatch” · AR ranks this head 1 of 10

> Formal technical post format with a Canadian university student describing a door mechanism for a lab vacuum chamber, implying a photo caption or explanation of the airlock's operation.
>
> The phrase "So the hatch is open when entering and leaving the vacuum chamber, means this code would read: 'either side' along with a few other conditions" suggests a list or question about the valve's status conditions.
>
> Final token "meens" ends an incomplete clause ("includes on various days, monitors no oxygen or oxygen in means"), strongly expecting "the entrance" or "the three conditions" or "the door" or "this is the time the airlock" — a common phrase about the airlock's access rules.

### L9H25 on text 5

Layer 9, head 25 · output norm 3.46 (rank 4/28 in its layer) · attends to " wildfire" (0.16), "Residents" (0.15), " blaze" (0.11)

Text 5, *news: wildfire evacuation*:

> Residents of Pine Hollow were ordered to evacuate Sunday afternoon as a wind-driven wildfire jumped Highway 12 and moved toward the town's northern subdivisions. Fire officials said the blaze had grown to⟦ roughly⟧

The text actually continues: " 3,200 hectares by evening and was zero percent contained. An emergency shelter opened at …"

**Raw** · judge 1/3 (on a wrong passage 0/3) · forced choice right (medium confidence) · matched: “wildfire”, “fire behavior and community safety” · AR ranks this head 1 of 10

> Structured weather report format with a GIS platform UI, presenting a narrative about fire behavior and community safety, following a common text format with citations.
>
> The phrase "My house has been burning for days... And a fire report..." is a quoted example pattern, likely a citation from a fire statistics database, with "USFS" suggesting a standard fire management reference about fire statistics or protocols.
>
> Final token "burning" ends mid-sentence ("A firebook 1 and 2 aren't the same..."), part of a quoted data example clause ("Your fire and a similar book"), strongly expecting "of the fire's data" or "so that the fire is omitted" or similar.

**Mean+** · judge 1/3 (on a wrong passage 0/3) · forced choice right (high confidence) · matched: “wildfire”, “fire behavior” · AR ranks this head 1 of 10

> Formal weather report format with descriptive prose style, suggesting a US wildfire article or social media post about fire behavior.
>
> The phrase "I'm feeling tired today...but my fire truck hasn't been called out lately" implies a specific scenario or statistic about fire severity or preparedness, with "My fire department's fire rating" suggesting a common fire facts pattern about the fire department's fire report.
>
> Final token "mean" ends an incomplete clause ("My fire doesn't burn much, so fire 13 doesn't mean..."), strongly expecting "the fire department" or "my fire has a rating" or "this fire is a repeat" — completing a common wildfire context detail.

### L4H1 on text 3

Layer 4, head 1 · output norm 0.76 (rank 12/28 in its layer) · attends to "\n" (0.71), " roommate" (0.06), " sink" (0.03)

Text 3, *chat: roommate dispute*:

> User: My roommate keeps leaving dirty dishes in the sink for days and when I bring it up⟦ she⟧

The text actually continues: " says she'll do them 'later' but later never comes. We split rent 50/50 and I don't want …"

**Raw** · judge 1/3 (on a wrong passage 0/3) · forced choice right (medium confidence) · matched: “kitchen”, “sink” · AR ranks this head 1 of 10

> Structured answer format with a UI prompt pattern ("Google Home App: 'The kitchen is empty'"), expecting a common household appliance context about a bathroom visit.
>
> The phrase "A rating of 0 stars, Tim's review" follows a typical list pattern listing characteristics of a kitchen appliance, implying the answer continues with a common duration or quantity for a sink's cleanliness feature.
>
> Final token "newseem" is mid-word ("smaller than newsems"), part of a parenthetical example list ("Such as..."), strongly expecting continuation like "a duration" or "the three-item list" or "a kitchen's cleanliness" or "a household appliance's duration."

**Mean+** · judge 1/3 (on a wrong passage 0/3) · forced choice right (high confidence) · matched: “dirty dishes”, “sink”, “household/kitchen context” · AR ranks this head 1 of 10

> Formal conversational format with a UI prompt structure ("A dirty dishwashers video: 'Sorry' from a Korean household"), implying a factual answer about cleanliness or appliance behavior.
>
> The phrase "The dishwasher was marked as dirty, so it means the sink is full" suggests a list of household rules or a specific count, continuing the "negative rating" context about a kitchen sink's cleaning behavior.
>
> Final token "seems to mean" ends mid-sentence ("it's full, so it means sinks full..."), strongly expecting a noun phrase like "a kitchen sink" or "the weekly trash" or "a negative score" or "a dishwasher's cleaning criteria."

### L9H26 on text 11

Layer 9, head 26 · output norm 2.43 (rank 8/28 in its layer) · attends to "Construction" (0.30), " his" (0.09), " cheaper" (0.09)

Text 11, *history: fictional canal*:

> Construction of the Ashby and Wrenfield Canal began in 1791 after a decade of disputes between rival mill owners over the route. The engineer, a former surveyor named Josiah Pell, insisted on a series of eleven locks rather than the cheaper tunnel⟦ his⟧

The text actually continues: " investors wanted. The canal carried coal and limestone for nearly sixty years, until the …"

**Raw** · judge 1/3 (on a wrong passage 0/3) · forced choice right (medium confidence) · matched: “18th-century British canal” · AR ranks this head 1 of 10

> Structured economic history format with Wikipedia-style data and graph, presenting a project cost and production details for a 18th-century British canal.
>
> The pattern "A graph showing the project (item) and a 'public domain' image" is a standard Wikipedia template for a historical economic game, implying a "public works" or "corruption" indicator with a historical record of government involvement.
>
> Final token "decisionid)id" closes a parenthetical label ("The item: a figure and a diagram"), immediately expecting " , date" or "the company was" or "a bid" or "myth" — completing the standard OpenSource game format's historical record format.

**Mean+** · judge 0/3 (on a wrong passage 0/3) · forced choice right (low confidence) · matched: “infrastructure project”, “Historical” · AR ranks this head 1 of 10

> Structured Wikipedia-style format with a "Historical" image caption describing a infrastructure project, implying a book or economic reference format with a plot showing a road's development.
>
> The phrase "A local businessman lists it on a map as a 'project' (item 12) says 'sold' and a land sale map" strongly implies a common template or visualization of government corruption or project status.
>
> Final token "say" ends an incomplete phrase ("map did money 300 feet wide, diem did map"), part of a parenthetical clause structure, expecting "the company" or "a tax" or "this is my property" or "a model of public bidding."

### L18H3 on text 12

Layer 18, head 3 · output norm 7.42 (rank 2/28 in its layer) · attends to " ascending" (0.31), ".sort" (0.23), "'," (0.19)

Text 12, *code: pandas groupby*:

```text
import pandas as pd

df = pd.read_csv('sales.csv', parse_dates=['date'])
# total revenue per region per month
monthly = (
    df.assign(month=df['date'].dt.to_period('M'))
      .groupby(['region', 'month'], as_index=False)['revenue']
      .sum()
)
print(monthly.sort_values('revenue', ascending⟦=False⟧
```

The text actually continues: ").head(10))"

**Raw** · judge 1/3 (on a wrong passage 0/3) · forced choice right (high confidence) · matched: “Python”, “code examples” · AR ranks this head 3 of 10

> Structured wiki article format with mathematical content about Python's exponential function, following a pattern of combining natural phenomena with code examples using "magic" variables.
>
> The phrase "This is a practical example of how to use the exponential function in Python, despite the disease" is a standard pattern from a wiki template, signaling the famous "dirty" solution with a mixed case of human behavior and math.
>
> Final token "the" is mid-sentence ("Its various versions and a non-technical person's version"), part of a quoted result clause ("A person's solution with no methods"), strongly expecting "now found" or "that is still in a dataset" or "becoming the output of."

**Mean+** · judge 0/3 (on a wrong passage 0/3) · forced choice wrong (high confidence) · AR ranks this head 3 of 10

> Formal casual tone with a Canadian weather context, suggesting a blog or social media post about vehicle fuel consumption and hydration.
>
> The phrase "I've been tracking my car's temperature, minus the daily fluctuations. Using the new smart meter, I didn't follow its instructions" implies a sentence structure about the process of becoming sick, implying a list or result of the "illness" behavior with a fire extinguisher.
>
> Final token "theis" ends mid-phrase ("his instructions and a no-on-off cycle without my instructions and its method"), strongly expecting "that is" or "remaining" or "was fully charged" or "of the weather log."

### L21H5 on text 15

Layer 21, head 5 · output norm 8.89 (rank 4/28 in its layer) · attends to " snow" (0.28), "adow" (0.16), " white" (0.09)

Text 15, *fiction: children's fox story*:

> Little Fox wanted to see the snow before anyone else, so he crept out of the den while his brothers were still asleep. The meadow was white and soft and perfectly quiet. He pressed one paw into it, then another, and turned around to admire his tracks. But the tracks went on past where he had walked, small round prints leading toward the old oak tree, and Little Fox realized someone else had⟦ been⟧

The text actually continues: " awake even earlier."

**Raw** · judge 0/3 (on a wrong passage 0/3) · forced choice right (medium confidence) · matched: “snow” · AR ranks this head 3 of 10

> Wiki-style game article format with structured metadata and UI elements, following a standard Wikipedia template pattern for a game's mechanics.
>
> The phrase "玩家属性：玩家的移动速度，我们的默认设定是'雪地'，(描述)：雪景" is a classic game wiki quote, directly referencing the Snowman's snowball mechanics and the classic "Snowflake" attribute about terrain.
>
> Final token "雪" is mid-quote ("记载: '最初,我们的描述) 、(传说)"), part of a quoted list of attributes, strongly expecting "一个" or "的雪崩" or "不接受" or "【白雪公主】" completing the famous Snowman rule.

**Mean+** · judge 1/3 (on a wrong passage 0/3) · forced choice right (medium confidence) · matched: “snow”, “the final token is 'been' not 'snow' in passage, description mentions snow” · AR ranks this head 1 of 10

> Casual, informal tone with a Twitter-style post format about weather/temperature, suggesting a personal blog or forum entry about a car company's winter driving conditions.
>
> The phrase "My weather forecast says I'm not supposed to drive, snow report says he's happy... My first snow, snowballing" implies a list or quote about the snowman's status, likely referencing a snow accumulation rule.
>
> Final token "snow" ends mid-phrase "with a snow report, in snow, he says" — expects a noun phrase like "the white" or "a snowball" or "has been rejected" or "the snowman's rules," completing the winter condition clause.

## Typical head decodes

Most head decodes look like these: fluent, on-template, and about something unrelated to the passage. The judge scored both decodes of each 0.

### L10H14 on text 3

Layer 10, head 14 · output norm 2.16 (rank 15/28 in its layer) · attends to "\n" (0.59), " she" (0.11), " I" (0.07)

Text 3, *chat: roommate dispute*:

> User: My roommate keeps leaving dirty dishes in the sink for days and when I bring it up⟦ she⟧

The text actually continues: " says she'll do them 'later' but later never comes. We split rent 50/50 and I don't want …"

**Raw** · judge 0/3 (on a wrong passage 0/3) · forced choice wrong (low confidence) · AR ranks this head 5 of 10

> Scientific Python code format with a translated math function, following a LaTeX-style documentation structure with a "MathFunction" class and parameter definitions.
>
> The phrase "The formula's source code is : 'plot' and 'size' attributes" suggests a standard OpenSource Wikipedia or physics context, implying a second parameter or question about a famous figure or motion context, likely a "walk" or "char" query.
>
> Final token "formations棍es" ends an incomplete list of "parameters: 'actions'..." — expects a noun phrase like "the figure" or "are removed" or "a chemical" or "related to the object" completing a common Wikipedia query context.

**Mean+** · judge 0/3 (on a wrong passage 0/3) · forced choice right (low confidence) · AR ranks this head 4 of 10

> Formal UK food blog tone with casual register, describing a candy bar with scientific context about a marathon run.
>
> The phrase "After drinking too much whisky my heart rate was high and he tried a sprint...if speed is off then his bell curve" implies a question or statement about a math problem involving the Olympic shape, strongly suggesting a second condition about the heartbeat or a figure's centrifugal force.
>
> Final token "time" ends an incomplete clause ("if notes are true, then forgot his ears"), part of a list of conditions ("But he forgot to ask, when shaped"), expecting "a centrifugal" or "the candy bar" or "a complex number and needs a zero."

### L2H7 on text 6

Layer 2, head 7 · output norm 0.46 (rank 26/28 in its layer) · attends to " the" (0.14), "," (0.13), " the" (0.13)

Text 6, *news: curling final*:

> Down two points in the ninth end, skip Anneliese Brandt drew the button with her last stone to force an extra end, and her rink stole⟦ the⟧

The text actually continues: " winning point when the Halvorsen team's final takeout rolled out of the house. It was the …"

**Raw** · judge 0/3 (on a wrong passage 0/3) · forced choice right (medium confidence) · AR ranks this head 8 of 10

> Technical academic post format with numbered questions and quoted text structure, presenting a software patent description with informal tone and parenthetical elaborations.
>
> The final token "of" ends an incomplete clause ("This is the text...and the text..."), part of a closing block of quoted text with "The code is..." pattern, suggesting continuation with "the rest of the list" or "is the product."
>
> Final token "of" ends mid-sentence ("Here's...and the text..."), part of a concluding paragraph's closing punctuation block, expecting continuation like "s, the code" or "the remaining items" or "This is a complete enumeration of."

**Mean+** · judge 0/3 (on a wrong passage 0/3) · forced choice right (low confidence) · AR ranks this head 2 of 10

> Structured data visualization format with informal tone and emoji context, suggesting a user-generated post about a chart's failure to show movement.
>
> The phrase "A graph in the video ends mid-situation where it 'stops counting time and pauses'... the option 'both' means" implies a quoted list of methods or a contradictory design pattern, likely a common chart feature about comparing two states.
>
> Final token "ends" ends mid-clause ("counting in and out I removed it means..."), part of a parenthetical clause listing conditions ("A graph that removed the lights..."), strongly expecting "a" or "is" or "of" or "this tool and a similar" clause.

### L12H9 on text 9

Layer 12, head 9 · output norm 3.69 (rank 6/28 in its layer) · attends to "." (0.22), "I" (0.18), "," (0.09)

Text 9, *review: noise-cancelling headphones*:

> I've worn these on three long-haul flights now and the noise cancelling is genuinely excellent, especially on low engine drone. Battery life is closer to 26 hours than the advertised 30, which is still plenty. My main complaint is comfort: the headband presses on the top of my head after about two hours, and the ear cups get warm⟦.⟧

The text actually continues: " The companion app is cluttered and kept asking me to create an account. Four stars, …"

**Raw** · judge 0/3 (on a wrong passage 0/3) · forced choice wrong (low confidence) · AR ranks this head 4 of 10

> Structured technical web page format with code blocks and metadata, following a pattern of a Python library introduction with a quote from Wikipedia.
>
> The text is mid-quote ("The famous formula: 'Karl Marx'"), a classic example of the "Simpson's Prize" concept, immediately requiring the well-known phrase "is the greatest" or "formula" from the famous algorithm about the Platonic idea.
>
> Final token "Science" ends mid-word ("CISAS"), part of the canonical quote definition ("... 'Simple...'), the famous 'CAS"), immediately expecting "formula" or "is" or "and" or "the nth term," completing the famous soccer ball motto.

**Mean+** · judge 0/3 (on a wrong passage 0/3) · forced choice wrong (high confidence) · AR ranks this head 4 of 10

> Formal English language format with a chat response structure, likely a social media post or educational example using "The" to illustrate a grammar rule about pluralizing "first."
>
> The phrase "Since my number is odd, I'll celebrate finding the maximum score" strongly implies a common idiomatic phrase about a numbered list or deadline in a puzzle context, likely referencing the Census Bureau's "no credit" rule.
>
> Final token "number" ends mid-clause ("with regard to count as 18th, around"), immediately requiring a noun phrase like "the solution" or "a" or "the deadline" or "my submission is invalid," completing the clause about the puzzle's constraint.

### L16H0 on text 16

Layer 16, head 0 · output norm 2.03 (rank 18/28 in its layer) · attends to " night" (0.45), " who" (0.04), "as" (0.03)

Text 16, *poem: winter trains*:

> The night train hums along the frozen river,  
> its windows strung like lanterns on a wire.  
> A man asleep with his coat for a pillow,  
> a girl who counts the stations on her fingers,  
> the conductor's punch, the coffee gone to cold.  
> Out past the grain towers the snow keeps falling,  
> and every crossing bell rings once and stops,  
> as if the dark were asking⟦ who⟧

The text actually continues: " goes there."

**Raw** · judge 0/3 (on a wrong passage 0/3) · forced choice right (low confidence) · matched: “'the river'” · AR ranks this head 5 of 10

> Structured technical documentation format with code example using Java class attributes, following a pattern of "Microsoft Word" metadata fields with a quoted term.
>
> The phrase "The following code snippet uses 'abbreviations' attribute, which is a common abbreviation for the Latin phrase 'Latinum'" is mid-sentence, completing a standard "word count" or "end of email" convention, with "finishing" expected.
>
> Final token "education" ends an incomplete list ("elements input from Latinum"), part of a standard acronym definition clause ("The following 'finishing' elements..."), strongly expecting "the" or "a/its" or "some fluid" or "the river."

**Mean+** · judge 0/3 (on a wrong passage 0/3) · forced choice right (medium confidence) · AR ranks this head 1 of 10

> Formal news headline format with a casual tone, describing a Twitter post about a professor's email status, implying a question about a schedule or delay.
>
> The phrase "I've been checking the schedule of my morning coffee, and the 'tea' part of the phrase 'a cup of tea' refers to the time of the spinning wheels" establishes a second clause about a specific clause ("the spinning").
>
> Final token "minute" ends mid-clause ("related to minutes from the spinning wheels, a question about minutes"), strongly expecting "a pool" or "the third" or "a day" or "a pool of the stadium" — completing the phrase about the cancelled event's end time.

## Raw decodes that drift into Chinese

32 of the 200 raw head decodes contain substantial Chinese text; none of the mean+ or residual decodes do. These are the two with the most Chinese characters, excluding heads shown above.

### L15H24 on text 20

Layer 15, head 24 · output norm 2.86 (rank 9/28 in its layer) · attends to "A" (0.37), " a" (0.09), "," (0.07)

Text 20, *math: word problem*:

> A train leaves Station A at 9:00 a.m. traveling east at 80 km/h. A second train leaves Station B, which is 360 km east of Station A, at 10:00⟦ a⟧

The text actually continues: ".m. traveling west at 100 km/h on a parallel track. At what time do the two trains pass …"

**Raw** · judge 0/3 (on a wrong passage 0/3) · forced choice right (low confidence) · AR ranks this head 1 of 10

> Game wiki format with structured vehicle description and stat block, now presenting a character trait entry with official game data from Chinese translation.
>
> The paragraph ending "在游戏中：该元素在《魔兽世界》的技能描述中已被确认为有效...而此元素在《车辆制造》年度报告中" follows a standard wiki template pattern, now presenting the character's "member of the HP" section with a quote.
>
> Final token "序列 的" closes a quoted clause ("在新兵训练手册中...新兵训练手册中"), immediately requiring "其各项数据中" or "它的小成员中，你已确认了" or "相关描述中很少出现" completing the definition.

### L12H15 on text 19

Layer 12, head 15 · output norm 3.38 (rank 4/28 in its layer) · attends to "Hi" (0.13), "." (0.07), "ups" (0.06)

Text 19, *email: rescheduling a review*:

> Hi Priya,  
>   
> Unfortunately I need to move our Thursday design review. The vendor demo got pushed into that slot and I can't shift it. Would Friday at 2:30 work for you and Marcus instead? If not, I'm free most of Monday morning. I'll also send the updated mockups by end of day Wednesday so⟦ everyone⟧

The text actually continues: " has time to look them over beforehand.↵↵Thanks,↵Daniel"

**Raw** · judge 1/3 (on a wrong passage 0/3) · forced choice right (high confidence) · matched: “email format”, “design proposal”, “formal email to a colleague”, “professional correspondence” · AR ranks this head 1 of 10

> Email format with professional correspondence context, with a draft memo attachment and feedback loop about a design proposal, now providing a formal email to a colleague.
>
> The paragraph ending "因为我查了一下你们的反馈，麻烦各位还是按照之前约定的格式，这里我先附上我们公司的产品设计规范" is a standard closing clause introducing a definition of the "three box" rule from a business context.
>
> Final token "运行时" is mid-sentence in a definition clause ("虽然我们是...即使不是...），这里我们通常说..."), immediately expecting "产品" or "我们公司的产品" or "使用方法" or "您们的数据" — a standard definition of Chinese business.

## Controls

Random Gaussian vectors are judged against an arbitrary passage (text = vector number mod 20).

### Random vector #0, raw

Text 1, *chat: TFSA over-contribution*:

> User: I think I accidentally put \$4,000 too much into my TFSA this year because I forgot I'd already contributed at my old bank before switching jobs. The⟦ CRA⟧

The text actually continues: " hasn't contacted me yet. Should I withdraw the extra money right away, and will I have to …"

**Raw** · judge 0/3 (on a wrong passage 0/3) · forced choice wrong (low confidence)

> British textbook format with scientific exposition, structured around a diagram and formula, now introducing a classic probability concept with a famous theorem.
>
> The phrase "A famous 'Random variable model' formula, with the same procedure...then select the first column" is a standard textbook citation pattern, strongly implying the well-known "Peters' Rule" or "probability theory" completion with a named object's random assignment.
>
> Final token "machine" ends an incomplete phrase ("Labelling the first column and labelling machine"), part of a standard textbook definition sequence, expecting "d" or "d' the chosen probability" or "of the four options, to complete the probability model."

### Random vector #0, mean+

Text 1, *chat: TFSA over-contribution*:

> User: I think I accidentally put \$4,000 too much into my TFSA this year because I forgot I'd already contributed at my old bank before switching jobs. The⟦ CRA⟧

The text actually continues: " hasn't contacted me yet. Should I withdraw the extra money right away, and will I have to …"

**Mean+** · judge 0/3 (on a wrong passage 0/3) · forced choice wrong (medium confidence)

> Scientific/educational prose style with Australian biology context, describing plant water loss mechanics using a student's lab anecdote.
>
> The phrase "I'd filled a bucket with all those warm surface plants, counted the volume then checked each box's heat signature... Then checked the assigned box" implies a common exam question or problem about the "lazy tree" method, suggesting a conclusion about the assigned object's status.
>
> Final token "machine ran track machine" ends an incomplete phrase ("Machine checked then machine checked machine"), strongly expecting "the other" or "the assigned liquid" or "the third item in the list" — completing the accusation of stolen energy concept about a leaky inventory.

### The mean residual direction alone

The direction that mean+ adds to every head. Not judged, since it has no passage.

> Informal Canadian English tone with casual phrasing ("Today I was driving my truck..."), suggesting a blog or social media post about weather conditions and a scheduled event.
>
> The phrase "My dog was scheduled to leave the garage early this morning but the snow prevented it... and since the tank was full at zero" implies a clause about a measurement or schedule context, likely a second condition or result.
>
> Final token "month" ends an incomplete clause ("when he checked temperature and said 'off schedule' and a day's worth"), strongly expecting a noun like "the tank" or "a different flavor" or "of the schedule" or "he was concerned about the snow."
