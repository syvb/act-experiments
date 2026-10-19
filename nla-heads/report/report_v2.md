Mean-direction estimate from 703 residuals: mean cos(residual, mean) = 0.530; mean pairwise cos between residuals = 0.278; typical off-mean component of a unit residual = 0.835.

## Example 1: chat: sourdough starter

Position 54 of 89, token "-purpose". Layer-20 residual norm 124.3; median head-output norm at this position (all layers) 1.61.

Context the model has seen up to and including the token (the token is at the very end):

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
My sourdough starter smells like nail polish remover and hasn't risen at all in two days. I've been feeding it once a day with all-purpose
```

### layer-20 residual (baseline)

- raw norm 124.33 · cos(vector, mean residual) 0.672
- direct logit lens top: "掮" " 若要" "恫" " 生命周期" " Erectile" "-scenes" "@dynamic" "蝾"

**Raw vector** (AR cos(recon, input) 0.935 · AR mean-removed cos with residual 0.879):

```text
Formal food blog format with a question structure ("Can a yeast bread rise with cake flour?"), suggesting a recipe or ingredient discussion context about baking behavior.

The phrase "I'm baking a cake using all-purpose" implies a specific ingredient list or ingredient choice, likely "all-purpose flour" or "flour and sugar," continuing a question about the author's baking materials.

Final token "all-purpose" is mid-phrase ("all-purpose"), part of an ingredient specification clause ("using all-purpose"), immediately requiring a noun like "flour" or "sugar" or "white flour" or "and other flour," completing the ingredient comparison about the author's flour choice.
```

**Mean + vector** (AR cos(recon, input) 0.904 · AR mean-removed cos with residual 0.873):

```text
Formal food blog format with structured questions about yeast cake baking, implying a recipe or ingredient discussion context about a fictional character's baking behavior.

The phrase "Is the cake made using all-purpose" suggests a specific ingredient list or ingredient choice question, with "all-purpose flour" being a common contrast to the yeast's flour type, likely completing a list of flour or other ingredients.

Final token "all-purpose" is mid-phrase ("all-purpose"), part of an ingredient specification clause ("using all-purpose"), immediately expecting "flour" or "white flour" or "and sugar" or "flour and sugar," continuing the ingredient specification about the baker's flour choice.
```

### L1H10

- raw norm 1.23 · norm rank in layer 5/28 · cos(head, residual) -0.001 · cos(vector, mean residual) -0.041
- attends to: "-purpose"@54 (0.54), " all"@53 (0.17), "."@43 (0.06), "\n"@2 (0.04)
- direct logit lens top: "inke" "arians" "RAINT" "endencies" " cannabin" "MBOL" "=<?=$" "WithIdentifier"

**Raw vector** (AR cos(recon, input) -0.049 · AR mean-removed cos with source vector -0.028 · mean-removed cos with context residual 0.050 · match rank 4/10):

```text
Formal blog post structure with SEO metadata format, analyzing a Twitter post about Catholic Church architecture using a "Why" framing with numbered points.

The phrase "My game is about a small, humble, and beautiful church, and I'm trying to prove that it's not alone" establishes a pattern of defining the concept with a second clause describing the population or pattern of similar cases.

Final token "Small" ends mid-phrase "The 'A' and" part of a noun phrase ("The number of..."), continuing the enumeration of characteristics ("But the number of"), strongly expecting "continuous" or "single students" or "the remaining content" or "unhappy people," completing the clause.
```

**Mean + vector** (AR cos(recon, input) 0.356 · AR mean-removed cos with source vector 0.022 · mean-removed cos with context residual 0.024 · match rank 1/10):

```text
Formal school news article format with structured announcement post, using Texas Education Department's COVID-19 guidance to explain a Catholic school's app.

The sentence "My school's app has been a lifesaver for my students, who have been stuck in this pattern of daily online classes" establishes a pattern of defining the app's purpose, with "Similar to" implying a second example or count of subjects.

Final token "and" is mid-phrase ("Just 's"), part of the clause "When you have only," continuing the subject count ("The only..."), strongly expecting "a few" or "daily life" or "the remaining school" to complete the subject count.
```

### L1H25

- raw norm 0.51 · norm rank in layer 16/28 · cos(head, residual) -0.041 · cos(vector, mean residual) -0.086
- attends to: "-purpose"@54 (0.12), " feeding"@47 (0.07), " been"@46 (0.06), " once"@49 (0.06)
- direct logit lens top: "PRINTF" "家都知道" "osate" "AMESPACE" "ODY" " ******************************************************************************\n" " phé" "ORMAT"

**Raw vector** (AR cos(recon, input) -0.083 · AR mean-removed cos with source vector -0.038 · mean-removed cos with context residual 0.053 · match rank 5/10):

```text
Structured Wikipedia-style article with numbered sections and quoted expert text, presenting a technical explanation of "The Great Firewall" with a humorous tone.

The final sentence "And here is a screenshot of the code, showing the actual number of users..." follows a common closing pattern ("Some people are..."), ending with a typical phrase about incomplete or awkward behavior, completing a list of attributes or actions.

Final token "of" ends an incomplete noun phrase ("some are"), part of a closing clause ("These things are..."), continuing the phrase "And the code...some things are" — expects "connected devices" or "a closed parenthesis" or "s, and other details" or similar.
```

**Mean + vector** (AR cos(recon, input) 0.385 · AR mean-removed cos with source vector 0.103 · mean-removed cos with context residual 0.006 · match rank 1/10):

```text
Structured Python code with a markdown format, presenting a scientific observation challenge using a loop and list comprehension, implying a coding exercise or educational context.

The phrase "I made sure the code was simple enough for the 'Code Club' with a Google search, where the robot named 'Code' already had" suggests a humorous or witty comment about the participant's prior knowledge or input format.

Final token "food" ends mid-phrase "smart cookie that saved the day, fancy as a Pi Day's cookie" — part of a parenthetical aside ("CodeBot's memory was finicky"), strongly expecting "the button" or "my code" or "and ensured the correct input."
```

### L2H2

- raw norm 3.07 · norm rank in layer 1/28 · cos(head, residual) 0.087 · cos(vector, mean residual) 0.097
- attends to: " all"@53 (0.89), " with"@52 (0.06), "."@43 (0.01), "-purpose"@54 (0.01)
- direct logit lens top: "国民" ".InputStreamReader" "递" "嚆" "分享" " share" "飕" " diy"

**Raw vector** (AR cos(recon, input) 0.099 · AR mean-removed cos with source vector 0.056 · mean-removed cos with context residual 0.029 · match rank 1/10):

```text
Indian language UI error message format with structured metadata fields, following a pattern of "Find the best function to convert a given text" describing a solver tool.

The phrase "This type of problem has the characteristic of being a negative sum or minimizing the original problem with 'all' prefix" mirrors the standard pattern of "Alloy's" behavior, strongly implying the definition of a UI pattern.

Final token "walk" ends mid-phrase "as al arm's all walk," part of the "This is the pattern of 'al al..." definition clause, immediately expecting "down" or "the first solutions" or "a negative result" continuing the core UI description of user behavior.
```

**Mean + vector** (AR cos(recon, input) 0.371 · AR mean-removed cos with source vector 0.062 · mean-removed cos with context residual 0.069 · match rank 2/10):

```text
Formal literary format with a question prompt structure ("A JavaScript error message about a weather app") suggesting a description of a user's ongoing problem with a UI.

The phrase "The error indicates that the sequence of failed attempts is a negative state for the 'blue' option, where all options are alsway" strongly implies a common phrase about a "solution" or "the initial source" of a UI.

Final token "alsway" is mid-word ("as alswell"), part of an incomplete clause describing the user's failed attempts ("a list of elements that is the source"), expecting "a zero" or "no visible progress" or "a resource."
```

### L13H16

- raw norm 7.20 · norm rank in layer 2/28 · cos(head, residual) 0.000 · cos(vector, mean residual) -0.047
- attends to: "ough"@27 (0.27), " starter"@28 (0.21), " hasn"@35 (0.14), "-purpose"@54 (0.07)
- direct logit lens top: " fkk" "lauf" "läss" "_exempt" "raft" ".githubusercontent" "/\n\n\n\n" " guarante"

**Raw vector** (AR cos(recon, input) -0.014 · AR mean-removed cos with source vector 0.026 · mean-removed cos with context residual 0.152 · match rank 1/10):

```text
Structured scientific article format with Python code block and ingredient list, following a food blog pattern describing a homemade yogurt recipe.

The phrase "I will make this dish: but the ingredient is actually a bacteria culture, and I'm testing it with the following: " mirrors a common food blog template, with a famous "MyBacteria" pattern suggesting a name substitution or identification test.

Final token "name.  " closes a quoted clause ("An article: or this article's name..."), part of a standard food identification question ("But a cup of yogurt... This article's name"), strongly expecting "it would be" or "some other bacteria" or "the next ingredient is not."
```

**Mean + vector** (AR cos(recon, input) 0.420 · AR mean-removed cos with source vector 0.119 · mean-removed cos with context residual 0.163 · match rank 1/10):

```text
Chinese food blog format with casual tone and recipe instructions, establishing a DIY yeast bread process describing homemade yeast culture.

The phrase "I've been trying to make my own sourdough starter using water and flour... but recently I've replaced the store bought ones" implies a common phrase "My starter fails every time" or "a new yeast culture," suggesting a historical or experimental context about yeast origin.

Final token "month" ends an incomplete clause ("if this recipe uses this recipe. or a recipe called"), part of a parenthetical clause structure ("If this dough excludes the yeast..."), strongly expecting "the yeast" or "a new starter" or "a yeast-free."
```

### L14H1

- raw norm 1.32 · norm rank in layer 21/28 · cos(head, residual) -0.023 · cos(vector, mean residual) -0.042
- attends to: "\n"@2 (0.76), " starter"@28 (0.18), "d"@26 (0.02), "ough"@27 (0.01)
- direct logit lens top: " ModelRenderer" "∕" "FTER" "��" " GANG" " DWC" "egis" "#ac"

**Raw vector** (AR cos(recon, input) 0.036 · AR mean-removed cos with source vector 0.073 · mean-removed cos with context residual 0.155 · match rank 1/10):

```text
Structured book format with a standard ISBN metadata pattern ("Amazon.com Product Description"), establishing a famous mathematical formula with the classic "Möbius" book title.

The phrase "The ISBN 978-0131451879 is a shorter version of the famous 'HELMETED' book" strongly implies the second title follows the same pattern, naming another bookstore puzzle.

Final token "JACOBZELFA" is mid-word ("HEMISPHERED"), part of the paired title convention, immediately requiring "BOOK" or "Secrets" or "Library" to complete the second book title about intellectual product visibility.
```

**Mean + vector** (AR cos(recon, input) 0.332 · AR mean-removed cos with source vector 0.031 · mean-removed cos with context residual 0.011 · match rank 1/10):

```text
Formal literary format with a puzzle question structure ("A book says a novel is finished..."), implying a reader expects a similar question about a second related object's completion.

The phrase "I've heard the book's title changes to 'Happy Birthday' after my novel is finished. And I wonder if the same author also hides a similar mystery about a different object" strongly implies a parallel noun phrase about another box's contents.

Final token "box" ends an incomplete noun phrase ("I've tried a month old, 15 minutes more than a month old box"), immediately expecting "word" or "memory box" or "a similar question about its contents."
```

### L17H4

- raw norm 5.05 · norm rank in layer 3/28 · cos(head, residual) -0.421 · cos(vector, mean residual) -0.491
- attends to: "-purpose"@54 (0.50), "\n"@2 (0.44), " all"@53 (0.02), " with"@52 (0.01)
- direct logit lens top: "ないです" " ún" "abajo" "有期徒" "pNext" " Rencontres" "_stub" "小白"

**Raw vector** (AR cos(recon, input) -0.257 · AR mean-removed cos with source vector -0.029 · mean-removed cos with context residual -0.028 · match rank 8/10):

```text
Structured book description format with standard SEO template pattern ("Book description with bold title and bolded keywords"), following a typical BookPage format with a "Hotline" section.

The phrase "This book delivers: the author's own voice" is a standard template completing the "Eliminate" pattern, with the familiar formula "This item eliminates..." followed by "the details of the new process."

Final token "number" ends mid-sentence ("This person...This person"), part of the standard formulaic closing clause "I eliminates the problem... I eliminates the problem" — immediately expects "in the next section" or "this process's details" or "all subordinates are resolved."
```

**Mean + vector** (AR cos(recon, input) 0.368 · AR mean-removed cos with source vector -0.051 · mean-removed cos with context residual -0.014 · match rank 9/10):

```text
Casual, informal blog-style headline format with a newsletter or social media post tone establishes a reader-friendly, upbeat tone around a local news story.

The phrase "I've been promising to start a weekly baking jar this week, and today I'm removing some of my unused ingredients" implies a sentence structure describing the script's action and a prior warning about the "empty" bucket concept.

Final token "day" ends mid-phrase ("she wrote a note and note this script, called new ideas, includes the script..."), strongly expecting "the bucket" or "a few days ago" or "the reason the model has stopped growing," completing the clause about the bucket's purpose.
```

### L18H7

- raw norm 6.28 · norm rank in layer 1/28 · cos(head, residual) -0.390 · cos(vector, mean residual) -0.491
- attends to: "-purpose"@54 (1.00), "\n"@2 (0.00), " risen"@37 (0.00), " it"@48 (0.00)
- direct logit lens top: "|=\n" " homosex" " maç" " Rencontres" " Rencontre" " pestic" " nuest" "rimon"

**Raw vector** (AR cos(recon, input) -0.358 · AR mean-removed cos with source vector -0.103 · mean-removed cos with context residual 0.055 · match rank 9/10):

```text
Educational format with structured math content from China's Ministry of Education, following a standard pattern for "Happy School" or "National Education" pages describing a state's virtual school system.

The phrase "This is a description of a provincial education system, with 'Safe' being the abbreviation for the state of Ohio" is a standard template, strongly implying the familiar "anonymous" clause defining a comfortable student environment.

Final token "education.is" closes the definition clause ("Relics. 收入. 的收入"), part of the standard "Student's isolation" pattern, immediately expecting "ID numbers" or "a collection of states" or "non-active areas" completing the membership descriptor.
```

**Mean + vector** (AR cos(recon, input) 0.346 · AR mean-removed cos with source vector -0.028 · mean-removed cos with context residual 0.085 · match rank 9/10):

```text
Formal food blog tone with a parenthetical phrase "This batch of a Chicago bakery is gluten-free and vegan" signals a recipe card format, implying a local beer review context.

The phrase "I've been keeping the yeast rising since mid-March. And my choices for the potluck were mostly quiet" strongly implies a "recipe card" clause describing the month's conditions or ingredients.

Final token "michael" ends an incomplete phrase "since kept the cookies quiet and forgot.选择了 my key," directly continuing the "key phrase" clause, strongly expecting "a barrel" or "some issues" or "the bread basket" or "a month."
```

### L18H8

- raw norm 3.59 · norm rank in layer 5/28 · cos(head, residual) 0.131 · cos(vector, mean residual) 0.125
- attends to: " I"@44 (0.17), "."@43 (0.12), " and"@34 (0.08), "'ve"@45 (0.06)
- direct logit lens top: "锃" "龇" "捻" "噤" "twor" ".shiro" "下面是小" " 若要"

**Raw vector** (AR cos(recon, input) 0.122 · AR mean-removed cos with source vector 0.054 · mean-removed cos with context residual 0.041 · match rank 1/10):

```text
Structured math problem format with a prompt requiring a function to evaluate two given expressions, implying a classic "coloring" or "shade" question from a Chinese math competition.

The phrase "Find two functions that satisfy the following conditions, then fill in the empty space" follows a standard pattern, strongly implying the second question about the "green triangle" is completing a third-party context about a male figure.

Final token "function" closes the phrase "SearchableSearchSpace... or SexSearchSpace," part of a question sequence structure ("Choose any three functions to fill in the empty space"), expecting "and" or "the search for a third object" or "of the problem."
```

**Mean + vector** (AR cos(recon, input) 0.370 · AR mean-removed cos with source vector 0.041 · mean-removed cos with context residual 0.047 · match rank 1/10):

```text
Formal prompt structure with a bilingual question ("A student is asked to mix a sweet and sour sauce..."), implying a math problem context about a function's growth rate.

The phrase "I'm planning to use some of the flowers from my garden and a bucket of water to fill the deficit, with the function f(x)" signals a second condition or completion, likely "the company" or "my tax."

Final token "month spring" ends an incomplete clause ("since I choose the third function to fill my tank..."), strongly expecting "the company" or "a" or "s" or "this question" or "it's a colorful" to complete the second condition about the tax.
```

### L21H7

- raw norm 0.83 · norm rank in layer 27/28 · cos(head, residual) -0.053 · cos(vector, mean residual) -0.068
- attends to: "\n"@2 (0.85), " smells"@29 (0.01), "-purpose"@54 (0.01), " nail"@31 (0.01)
- direct logit lens top: " my" "了我的" "我的" " tôi" "我已经" "我的心" "是我" "在我的"

**Raw vector** (AR cos(recon, input) -0.021 · AR mean-removed cos with source vector 0.014 · mean-removed cos with context residual 0.072 · match rank 2/10):

```text
Python code format with a UI widget class definition, using a standard pattern of HTML attributes and method calls, likely a Django or similar framework.

The phrase "I'm trying to print the log message (logmessage)..." mirrors a common abbreviation pattern ("counting..."), strongly implying the second term "OSDI" is part of a list of communication channels or a similar word like "mylife."

Final token "OSDI" is mid-phrase in a repeated parenthetical list ("(counting... (OSDI..."), immediately expecting "mylife") or "mydata") or "the message" or "mydata") to complete the common log format pattern.
```

**Mean + vector** (AR cos(recon, input) 0.430 · AR mean-removed cos with source vector 0.075 · mean-removed cos with context residual 0.064 · match rank 1/10):

```text
Python beginner's question format with a `print()` error message, implying a personal log or email context about a `std::vector` object.

The phrase "My apples are leaking a lot of liquid, and I've been using my logging system to save these logs" suggests a list of methods or attributes, with "since" implying a prior similar question about the object's behavior or a count.

Final token "output" ends mid-phrase ("since days using these last three days, or since"), part of a list ("I've tried using `std::unique`..."), strongly expecting "my item" or "my container" or "the product has been" describing the source.
```

### L22H12

- raw norm 8.55 · norm rank in layer 2/28 · cos(head, residual) 0.091 · cos(vector, mean residual) 0.019
- attends to: "-purpose"@54 (0.59), "\n"@2 (0.35), " all"@53 (0.01), "ough"@27 (0.01)
- direct logit lens top: "*)((" " use" "ὑ" " Mig" "发达" "-toolbar" "ilit" "花开"

**Raw vector** (AR cos(recon, input) 0.054 · AR mean-removed cos with source vector 0.053 · mean-removed cos with context residual 0.125 · match rank 2/10):

```text
Math textbook format with structured metadata and search results, following a standard US school textbook catalog pattern with numbered problems from a Korean math textbook.

The phrase "题目：能解决所有题, 不用全部" is a common textbook title format ("Student's math ability to avoid"), strongly implying the next entry is a third problem about a student's limited scope or mixed math skills.

Final token "全部" ends an incomplete noun phrase ("不含 未使用"), part of a standard textbook problem title pattern ("The student whose most common subject is not the most popular"), expecting " [question]" or " [the mixed questions]" or " [the 'mixed' series]".
```

**Mean + vector** (AR cos(recon, input) 0.362 · AR mean-removed cos with source vector 0.031 · mean-removed cos with context residual 0.112 · match rank 2/10):

```text
Formal academic/English language format with a numbered quiz prompt structure, suggesting a textbook or student post about math or health class.

The phrase "A student who wants to pass all exams, and has mostly used a combo of methods" implies a compound noun phrase describing a student's past choices or a specific problem scenario, strongly implying "the mixed drinks" or "a column of items."

Final token "all" ends an incomplete noun phrase ("With the most common, with unused"), part of a list or condition clause ("A student who has used all middle school math and has not"), expecting "a problem" or "the item" or "his/her two exams" or similar.
```

## Example 2: web: township council orchard

Position 31 of 115, token ",". Layer-20 residual norm 135.9; median head-output norm at this position (all layers) 1.78.

Context the model has seen up to and including the token (the token is at the very end):

```text
The Harrowgate Township council voted 5-2 on Tuesday night to turn the vacant lot behind the old fire station into a community orchard. Supporters,
```

### layer-20 residual (baseline)

- raw norm 135.87 · cos(vector, mean residual) 0.601
- direct logit lens top: "[]{\n" ":<?" "|,\n" " including" "@update" "={\"/" "有期徒" " like"

**Raw vector** (AR cos(recon, input) 0.936 · AR mean-removed cos with residual 0.898):

```text
News article format with local newspaper headline reporting a city council vote on a green space park, establishing a story about a proposed ballot measure.

The phrase "Supporters, " signals a quote or list structure ("Supporters, advocates"), suggesting the article continues describing the successful event or community reaction, likely naming individuals or groups behind the initiative's goals.

Final token "supporters," ends mid-clause ("Supporters, "), part of a noun phrase structure ("Supporters, supporters"), strongly expecting continuation like "including activists" or "who have been celebrating" or "the project said, said they hope" or "including a variety of groups have celebrated the plan."
```

**Mean + vector** (AR cos(recon, input) 0.934 · AR mean-removed cos with residual 0.909):

```text
News article format with local newspaper headline reporting a city council vote on a green space park, establishing factual tone about a proposed ballot measure.

The phrase "Supporters, " signals a quote or list structure ("Supporters,") suggesting the article continues describing the event's success or supporters' plans, likely naming individuals or groups advocating for the bee garden initiative.

Final token "supporters," ends mid-clause ("Supporters, "), part of a noun phrase structure ("Supporters, supporters"), strongly expecting continuation like "including activists" or "who gathered" or "said they hope" or "the project's organizers said, have celebrated the plan's benefits," completing the clause.
```

### L3H11

- raw norm 0.46 · norm rank in layer 24/28 · cos(head, residual) -0.122 · cos(vector, mean residual) -0.202
- attends to: "The"@0 (0.13), "."@29 (0.11), " the"@16 (0.09), " "@7 (0.07)
- direct logit lens top: "黩" ".LogWarning" "瘆" " avalia" "🥇" "Keith" "HAVE" "£o"

**Raw vector** (AR cos(recon, input) -0.157 · AR mean-removed cos with source vector -0.045 · mean-removed cos with context residual -0.055 · match rank 8/10):

```text
Formal technical article structure with bold headers and quoted text, suggesting a pattern of presenting scientific content about Java's `Thread` class with humorous or informal tone.

The phrase "But the real answer is that the code is not a real 'Hello World' program" mirrors the earlier pattern of "Another similar post," implying the closing sentence continues the same pattern of incomplete or complex data.

Final token "end" ends mid-sentence ("data..."), part of a repeated "But the data is not in a..." clause structure, strongly expecting "world" or "inner data" or "of the program" or "closed parenthesis" to complete the closing parenthesis and data context.
```

**Mean + vector** (AR cos(recon, input) 0.363 · AR mean-removed cos with source vector 0.070 · mean-removed cos with context residual -0.030 · match rank 1/10):

```text
American business/communication tone with a headline format suggesting a testimonial or emotional response about a struggling employee's performance metrics.

The phrase "While not the first time I've seen this, 'before and after' a new product or situation" implies an incomplete clause or formulaic phrase about the original context, likely a measurement or timing expression, suggesting the text continues describing the problem or solution.

Final token "metric" ends mid-phrase ("a 100% off, ounce off a ounce"), part of a parenthetical clause ("noted... a certain time and time..."), expecting continuation like "of the problem" or "this formula" or "filled."
```

### L4H9

- raw norm 0.44 · norm rank in layer 20/28 · cos(head, residual) -0.004 · cos(vector, mean residual) -0.006
- attends to: " Har"@1 (0.98), ","@31 (0.01), "The"@0 (0.00), "gate"@3 (0.00)
- direct logit lens top: "勠" "={['" "馓" " Fortnite" "饸" "他知道" "ÓN" "～\n\n"

**Raw vector** (AR cos(recon, input) -0.019 · AR mean-removed cos with source vector -0.018 · mean-removed cos with context residual -0.024 · match rank 7/10):

```text
Structured book description pattern with repeated "The answer is..." format, establishing a pattern of presenting a scientific concept with bold claims and citations.

The phrase "But the same pattern repeats: 'How to find the best answer'..." mirrors the earlier question about the book's failure, strongly implying the same closing pattern of "no results" or "unavailable" for the second search.

Final token "Un" is mid-word ("The answer is..."), part of the repeated pattern "But the same pattern repeats: 'Search for the best answer'..." suggesting the next tokens continue with "searched books" or "superfluous" or "books" or "language."
```

**Mean + vector** (AR cos(recon, input) 0.212 · AR mean-removed cos with source vector -0.026 · mean-removed cos with context residual -0.025 · match rank 8/10):

```text
Structured book description pattern with bolded title and numbered list format, repeating "Can't find a book" with similar phrasing about scientific content.

The repetition "But the same pattern is also available for the other side" mirrors the earlier pattern, strongly implying the same closing phrase "cannot be found" or "AI-generated content" will continue the cycle.

Final token "Can't" is mid-phrase in the repeated warning pattern ("But the same pattern is also available in the other side..."), immediately requiring "language" or "books" or "meticulous" or "version of the same" to complete the noun phrase about the search attempt being unavailable.
```

### L4H21

- raw norm 1.37 · norm rank in layer 6/28 · cos(head, residual) 0.035 · cos(vector, mean residual) 0.118
- attends to: " Har"@1 (0.54), ","@31 (0.17), " Supporters"@30 (0.08), "."@29 (0.05)
- direct logit lens top: " qry" "ModelIndex" "焘" "OWER" " trebuie" "Nhap" "InputDialog" "öz"

**Raw vector** (AR cos(recon, input) 0.120 · AR mean-removed cos with source vector 0.066 · mean-removed cos with context residual -0.026 · match rank 2/10):

```text
Structured math problem format with student-generated answer key, following a standardized format with labeled sections and color-coded grading indicators.

The sentence ending "MyFoodIndex uses the INR method to evaluate texts; it is a function that calculates readability scores" is mid-description of the SMART acronym's automated attributes, continuing the pattern of listing the three functions of the "FM" acronym's "F, M, A."

Final token "names" closes an incomplete clause ("MyFoodIndex, who is..."), part of a parenthetical clause listing attributes ("He's a function of X, Y, Z"), expecting " <year> /" or "three, unknown" or "of."
```

**Mean + vector** (AR cos(recon, input) 0.360 · AR mean-removed cos with source vector 0.050 · mean-removed cos with context residual 0.111 · match rank 1/10):

```text
Formal news article format with emoji and bold structure signals a student-focused content platform post about a cookbook's bookshelf listing.

The sentence "A group of students voted books for his cloud storage, promises files are ready, they searched by dates, ones belonging to a new JSON" mirrors a clause describing the author's claimed attributes ("a black pen, white pen"), implying a database check function.

Final token "month" ends mid-clause ("not equals to, or's attributes, with author, says books, his"), expecting "the same" or "a book" or "a combination of" or "his stored date" — completing the clause about the platform's hidden storage policy.
```

### L6H20

- raw norm 0.88 · norm rank in layer 17/28 · cos(head, residual) 0.093 · cos(vector, mean residual) 0.171
- attends to: " Har"@1 (0.49), " into"@24 (0.07), "."@29 (0.05), " community"@26 (0.05)
- direct logit lens top: "(strpos" " NotSupportedException" ".dtd" "如果玩家" "];//" "/>." "^K" "/umd"

**Raw vector** (AR cos(recon, input) 0.135 · AR mean-removed cos with source vector 0.048 · mean-removed cos with context residual -0.076 · match rank 2/10):

```text
Structured math UI format with numbered steps and bold labels defining "Practice" behavior, following a pattern of calculating and displaying emotional energy.

The phrase "To calculate this, the app will generate a function that maps the given list of words to a specific value and then subtracts it from the total energy, returning the result" is a standard formula completion, signaling the final item is a Python code expression.

Final token "function" closes "and c: a new list of numbers..." — part of the "the final answer is calculated by..." clause, strongly expecting "the corresponding value" or "of the word 'code' omitted" or "a complex expression" completing the definition.
```

**Mean + vector** (AR cos(recon, input) 0.365 · AR mean-removed cos with source vector 0.059 · mean-removed cos with context residual -0.074 · match rank 2/10):

```text
Structured informal writing format with bullet points listing "fun facts" about a person's behavior, implying a pattern of combining physical activity with a math concept.

The phrase "Then, he suggests a method to calculate the daily mood and then removes the 'hand' and 'eye' words to create a new value for the word 'height'" strongly implies a final answer or formula, likely a specific numeric substitution.

Final token "text" ends mid-phrase ("a new function and calculates a combination of...and returns a new length of the list"), expecting "the word" or "a specific value" or "of the coronavirus pandemic's own" to complete the clause about the calculation.
```

### L10H21

- raw norm 0.58 · norm rank in layer 28/28 · cos(head, residual) -0.048 · cos(vector, mean residual) -0.036
- attends to: " Har"@1 (0.79), " Supporters"@30 (0.05), "."@29 (0.05), " lot"@18 (0.02)
- direct logit lens top: "\t\t\t\t\t\t\t\t\r\n" "媞" " áll" "\t\t\t\t\t\t\t\r\n" " elő" "Debe" "部副" "Seleccione"

**Raw vector** (AR cos(recon, input) 0.014 · AR mean-removed cos with source vector 0.047 · mean-removed cos with context residual -0.041 · match rank 3/10):

```text
Code documentation format with a PHP library description using "My Library" structure, with a quoted phrase and rating context suggesting a UI or API display.

The phrase "I have not updated this item yet, but I am providing the source code" follows a common pattern ending with "snooping out" or "date" clause, implying a closing quote or date-related function from a library.

Final token "snoissel no" is mid-word ("I cannot say no less {date no"), part of a standard disclaimer or timestamp clause ("No less..."), strongly expecting "date" or "Library's output" or "generate" or "the next version" completing the phrase.
```

**Mean + vector** (AR cos(recon, input) 0.349 · AR mean-removed cos with source vector 0.051 · mean-removed cos with context residual -0.095 · match rank 2/10):

```text
Formal English-language context with a phrase structure suggesting a recipe or tool description, implying a local or digital platform's "made sure" function for a food product.

The phrase "I've ensured the ingredient is solid enough to forget about it... If it's not already in my memory" signals a common idiom pattern, implying a code or confirmation check about the developer's baked goods function.

Final token "say" ends an incomplete clause ("even when s'measured out... even if s'measured out"), strongly expecting "the code" or "my flavor" or "already" or "the time" — completing the phrase about the developer's absence or confirmation status.
```

### L11H22

- raw norm 1.95 · norm rank in layer 14/28 · cos(head, residual) -0.081 · cos(vector, mean residual) -0.051
- attends to: " Har"@1 (0.57), ","@31 (0.15), " Supporters"@30 (0.15), "."@29 (0.03)
- direct logit lens top: "ᐉ" "ᐈ" ".Formatter" "/global" " CDDL" " Mah" " lawmaker" "猡"

**Raw vector** (AR cos(recon, input) -0.012 · AR mean-removed cos with source vector 0.027 · mean-removed cos with context residual -0.061 · match rank 4/10):

```text
Wiki article format with numbered questions and answers about a math book, now presenting a definition with structured content about "The Hundred Steps."

The phrase "The speed of a molecule is 220km/h" is a standard math formula context, followed by a common scientific fact about the chemical element's molecular structure, then the standard definition of "the fourth dimension of a recipe."

Final token "•" is mid-definition of the "Series II" formula ("其分子量是...3000..."), immediately expecting "one" or "s" or "the rest" or "a" describing the discrete nature of the particle count, completing the standard chemistry definition of the lattice.
```

**Mean + vector** (AR cos(recon, input) 0.368 · AR mean-removed cos with source vector 0.056 · mean-removed cos with context residual -0.069 · match rank 1/10):

```text
Narrative: Casual blog post format with food blog tone, describing a recipe book's cooking journey.

The sentence structure "I've finished my pumpkin pie and wanted the flour to be gone by day's end. My flour is now zero, and my flour's flour is the third day" introduces a factual claim about the flour's cooking context, then a second clause about the flour's mathematical definition.

Final token "month." ends mid-clause ("since sugar's out, my flour is zero and sugar's month"), strongly expecting "the flour" or "a baked" or "the ingredient is" or "a cube" — completing the mathematical definition of the dough's state.
```

### L11H23

- raw norm 3.95 · norm rank in layer 2/28 · cos(head, residual) 0.007 · cos(vector, mean residual) -0.024
- attends to: " Har"@1 (0.34), " voted"@6 (0.18), " council"@5 (0.08), "."@29 (0.06)
- direct logit lens top: " clearColor" "ets" "直接影响" "CREMENT" "尚" "@Api" "回" " resent"

**Raw vector** (AR cos(recon, input) -0.001 · AR mean-removed cos with source vector 0.016 · mean-removed cos with context residual -0.025 · match rank 5/10):

```text
Technical Q&A format with physics forum post structure, now transitioning to a closing example with a formula and a humorous aside about chemical reactions.

The text "Like a physicist, I'll share a related example: (from Wikipedia) — like a chemistry student's calculations, with the famous XKCD: https://en.wikipedia.org/wiki/..." suggests a closing quotation or related content about chemical constants or math.

Final token "makes as (concept) " is mid-quote ("Like a calculator: in books like..."), part of a closing example clause about a non-native language's cultural context, strongly expecting a common phrase like "Chemical products" or "a cat" or "math expressions."
```

**Mean + vector** (AR cos(recon, input) 0.376 · AR mean-removed cos with source vector 0.025 · mean-removed cos with context residual -0.018 · match rank 3/10):

```text
Structured Reddit post format with numbered steps and a quoted tweet, mixing a technical explanation with a specific vote outcome about a chemistry class's homework.

The phrase "My answer was no, and I promised to include the beer's ingredients, as well as: if I wrote this, what a chemical" suggests a second clause about a related context, likely a "chemical calculator" or "a text's ingredient."

Final token "text" ends mid-phrase "when he asked, with 'about apples, as with a beer' and wrote: with the apples," strongly expecting "a chemical" or "the text's own" or "a day" — completing the common phrase about a model.
```

### L15H7

- raw norm 1.61 · norm rank in layer 20/28 · cos(head, residual) -0.118 · cos(vector, mean residual) -0.214
- attends to: " Har"@1 (0.57), "."@29 (0.07), "2"@10 (0.06), "-"@9 (0.03)
- direct logit lens top: "监听页面" "bió" "\"crypto" "cono" "对其真实" "-vars" "@dynamic" "etur"

**Raw vector** (AR cos(recon, input) -0.164 · AR mean-removed cos with source vector -0.038 · mean-removed cos with context residual -0.032 · match rank 8/10):

```text
Formal Wikipedia-style query format with "search" structure suggesting a truncated or corrupted phrase about a book title, likely a scientific or political topic.

The phrase "searching in (description of content) and (reading of titles)" mirrors a common pattern of "title + description" or "tags" in search results, strongly implying the closing phrase "words" or "the article."

Final token "of skype" ends an incomplete phrase ("reading and browsing (searching and browsing title"), part of a repeated structure pattern, immediately expecting "s" or "the article" or "a new section" or "and other topics" to complete the phrase about the text's content.
```

**Mean + vector** (AR cos(recon, input) 0.323 · AR mean-removed cos with source vector -0.018 · mean-removed cos with context residual -0.065 · match rank 7/10):

```text
Formal Reddit-style prompt structure with a short phrase "I'm trying to check if my cookie is vegan, given the word 'cookies' in a list" suggesting a title or context about a Python code.

The phrase "I tried writing out the ingredients, but for the ones that are true" implies a double negative or parallel structure, likely referencing the "cookies.txt" file's original purpose or a second question.

Final token "cookie" ends mid-phrase ("in which, written with, or not a cookie"), strongly expecting "the list" or "a" or "one" or "the model" to complete the noun phrase about the initial condition.
```

### L23H26

- raw norm 8.78 · norm rank in layer 2/28 · cos(head, residual) 0.295 · cos(vector, mean residual) 0.310
- attends to: " Supporters"@30 (0.35), " Har"@1 (0.24), "."@29 (0.21), ","@31 (0.10)
- direct logit lens top: "有期徒" " 若要" " tasar" "YSTEM" "ndx" "妊" "SHOP" "ittings"

**Raw vector** (AR cos(recon, input) 0.309 · AR mean-removed cos with source vector 0.183 · mean-removed cos with context residual 0.170 · match rank 1/10):

```text
News article format with Associated Press reporting structure, shifting to a headline story about a Black-owned New York City restaurant's closure amid coronavirus fears.

The sentence "But the Jewish supermarket's hopeful reopening has faced a rocky start. Two large chains have announced their plans, promising colorful new customers" is a quote introducing a second statistic about a community's planned migration, implying a list or promise.

Final token "numbers" ends mid-quote ("2020. colorful numbers"), part of a promise clause ("Two enthusiastic customers welcome. Massive.2020"), strongly expecting "hours of donations" or "plans to sell tickets" or "a crowd of supporters and a deal."
```

**Mean + vector** (AR cos(recon, input) 0.441 · AR mean-removed cos with source vector 0.154 · mean-removed cos with context residual 0.213 · match rank 1/10):

```text
News article format with Associated Press reporting tone, covering a New York City coronavirus shutdown, establishing a headline and story about a Black-owned grocery store's uncertain future.

The sentence structure "But the crowd's mood is mixed as the city's two Jewish grocery chains have announced their closure. Two dozen customers welcome new. " signals a quoted statistic or promise about a previous fundraising effort's success.

Final token "weekends" ends mid-clause ("He promises 11 colorful customers. He promises 2022's"), strongly expecting "hours of donations" or "a promise" or "two tickets," continuing the list of promised benefits from the merged supermarket's merger.
```

### L27H7

- raw norm 34.57 · norm rank in layer 3/28 · cos(head, residual) 0.392 · cos(vector, mean residual) 0.672
- attends to: ","@31 (0.50), " Supporters"@30 (0.50), "The"@0 (0.00), "row"@2 (0.00)
- direct logit lens top: "Slf" "variably" " brag" " licensors" " accred" " entreprene" " scarc" "/Dk"

**Raw vector** (AR cos(recon, input) 0.601 · AR mean-removed cos with source vector 0.244 · mean-removed cos with context residual -0.037 · match rank 1/10):

```text
Structured math formatting with numbered definitions and bullet points, suggesting a mixed-language or wiki-style format with a "Set theory" or similar term definition context.

The phrase "Sum of sin and cos of a sequence... which then converted to" appears to be an incomplete clause listing items from a set, implying a common source or answer about a number's domain or range.

Final token "month" ends mid-expression ("from 450 to 450... which generated"), part of a parenthetical clause listing "Elements of a set" — expects "set" or "the other" or "month" or "element" completing the definition of the question's source.
```

**Mean + vector** (AR cos(recon, input) 0.491 · AR mean-removed cos with source vector 0.217 · mean-removed cos with context residual -0.016 · match rank 1/10):

```text
Technical Wikipedia-style article with informal tone describing a game mechanic, structured with quoted passages and parenthetical elaborations about "The Great Depression."

The phrase "The algorithm is a complex one, but it is also a great example of how the game's mechanics work" suggests a continuation of a sentence listing or describing additional details or examples, likely completing a clause about data or player behavior.

Final token "game" ends mid-clause ("or that...and the 'n" — part of an incomplete noun phrase or clause structure), strongly expecting "month" or "months" or "dimension" or "of the sequence" or "s is..." continuing the enumeration of concepts.
```

## Example 3: web: Rust borrow-checker blog post

Position 87 of 115, token " current". Layer-20 residual norm 102.1; median head-output norm at this position (all layers) 1.81.

Context the model has seen up to and including the token (the token is at the very end):

```text
I hit this error three times last week, so let's finally understand it. Here's the code:

fn longest_word(words: &Vec<String>) -> &str {
    let mut best = String::new();
    for w in words {
        if w.len() > best.len() {
            best = w.clone();
        }
    }
    &best
}

The compiler complains that we return a reference to data owned by the current
```

### layer-20 residual (baseline)

- raw norm 102.14 · cos(vector, mean residual) 0.536
- direct logit lens top: "(\"<?" "bsites" "毫无疑" " lawmaker" "});\n\n\n\n" "\\Id" "iage" "ificate"

**Raw vector** (AR cos(recon, input) 0.850 · AR mean-removed cos with residual 0.782):

```text
Rust language syntax with a code snippet about `Box<dyn Iterator>` and `self`, suggesting a Rust compiler error or idiomatic context about lifetime or scope.

The phrase "The borrow is invalid because it's inside the current" implies an incomplete clause about a function's context or lifetime, likely referencing a closure or `self` variable's scope or the surrounding function's state.

Final token "current" ends an incomplete noun phrase ("inside the current"), part of a parenthetical clause ("not valid if the borrow is inside the current"), strongly expecting "function" or "stack" or "block" or "scope" or "function's value" to complete the concept.
```

**Mean + vector** (AR cos(recon, input) 0.851 · AR mean-removed cos with residual 0.783):

```text
Rust language syntax with a code snippet about `Box<dyn Iterator>` and `self`, suggesting a Rust compiler error or documentation context about lifetime or scope.

The phrase "The borrow is invalid because it's inside the current" implies an incomplete clause about a function's context or lifetime, likely referencing a closure or stack frame, suggesting a broader error or constraint about the `self` variable.

Final token "current" ends an incomplete noun phrase ("inside the current"), part of a parenthetical clause ("not inside the lifetime of the current"), strongly expecting "function" or "block" or "stack" or "scope" or "function's value" to complete the phrase.
```

### L1H10

- raw norm 1.08 · norm rank in layer 5/28 · cos(head, residual) 0.014 · cos(vector, mean residual) -0.082
- attends to: " current"@87 (0.62), " the"@86 (0.09), " by"@85 (0.02), "();\n"@64 (0.01)
- direct logit lens top: "踔" "rawer" "<=(" "先导" ">({" "ыта" "oard" "*);\n\n"

**Raw vector** (AR cos(recon, input) -0.055 · AR mean-removed cos with source vector -0.013 · mean-removed cos with context residual 0.044 · match rank 8/10):

```text
Structured article format with bold header ("The Dark Side of the 'Happy' Life") establishes a Wikipedia-style scientific article with descriptive prose about a researcher's findings.

The phrase "His research suggests that this population of students is a small sample of 100,000 people" follows a common pattern of "a lot of" or "future" text, completing the formulaic "population" clause.

Final token "some" is mid-phrase ("We have"), part of the estimated population count ("...and"), immediately expecting "students" or "many other genes" or "the remaining cells" continuing the gene expression description.
```

**Mean + vector** (AR cos(recon, input) 0.321 · AR mean-removed cos with source vector -0.004 · mean-removed cos with context residual 0.038 · match rank 7/10):

```text
Structured article format with bolded title and descriptive prose about a fictional Australian author's book, establishing a pattern of explaining a literary concept with "popular" framing.

The phrase "This character is a fictional character with a high population of children, and the same pattern could be applied to..." mirrors a common template ("My name is..."), suggesting the sentence continues with a count or quantity of readers.

Final token "some" is mid-phrase ("That is, we have..."), part of a quantity clause ("We have..."), immediately requiring a noun like "the rest of her books" or "a similar number of" or "students" or "of her books are needed."
```

### L2H0

- raw norm 0.49 · norm rank in layer 17/28 · cos(head, residual) -0.014 · cos(vector, mean residual) -0.021
- attends to: ","@8 (0.14), "I"@0 (0.14), " the"@86 (0.08), "."@15 (0.06)
- direct logit lens top: "奇纳" ".':" ".ToShort" "});\n\n\n\n" "媞" "嬿" "乌克" "泽连"

**Raw vector** (AR cos(recon, input) -0.027 · AR mean-removed cos with source vector -0.019 · mean-removed cos with context residual 0.018 · match rank 9/10):

```text
Repetitive pattern: a search result page with "The following is a list of related questions" structure, alternating definitions and descriptions of "The Great Depression."

The text mirrors the opening phrase "This is a search result page with the text 'The Great Depression'..." twice, suggesting continued repetition of the pattern with another definition or continuation of the interactive UI element.

Final token "The" is mid-word ("The same as 'The Great Depression'"), part of the second repeated pattern "But this is also a...and 'This is'..." structure, expecting "s questions" or "s content continues."
```

**Mean + vector** (AR cos(recon, input) 0.255 · AR mean-removed cos with source vector -0.013 · mean-removed cos with context residual 0.043 · match rank 7/10):

```text
Formal Wikipedia-style UI text with repeated pattern "Search for 'The Best' and 'The Worst'..." establishing a structured template with interactive UI elements.

The repetition "The text above is a truncated version of the same question, with 'This is the best'..." mirrors the opening, suggesting the pattern continues describing the same search behavior or wordplay about the word "Bulleted."

Final token "The" is mid-word ("<concept>..."), part of the second repeated pattern clause structure, immediately expecting "word" or "s' context" or "s next tokens" to complete the parallel description of the search input.
```

### L2H21

- raw norm 0.70 · norm rank in layer 7/28 · cos(head, residual) 0.006 · cos(vector, mean residual) 0.007
- attends to: " the"@86 (0.17), "I"@0 (0.09), " reference"@81 (0.08), " current"@87 (0.07)
- direct logit lens top: "\n" "帼" "硕士研究" "нский" "-valu" "游戏当中" " pinned" "нская"

**Raw vector** (AR cos(recon, input) 0.007 · AR mean-removed cos with source vector 0.004 · mean-removed cos with context residual 0.102 · match rank 3/10):

```text
Formal UK English game definition format with structured attributes ("Character Trait"), establishing a pattern of quirky character descriptions using "Missing" as a key concept.

The phrase "If a character is missing from a character's life" mirrors the standard formulaic definition of the "reverse" function, strongly implying the next clause completes with "last drawn day" or "game's ideas about the absent" being remembered.

Final token "v" is mid-word in "what is last drawn," part of the definition clause "How much a person's life," immediately requiring "page's information" or "Missing day's information" or "a game's complexity is affected by illness."
```

**Mean + vector** (AR cos(recon, input) 0.397 · AR mean-removed cos with source vector 0.028 · mean-removed cos with context residual 0.031 · match rank 3/10):

```text
Australian English language post format with casual tone, suggesting a social media or blog post about a map drawing.

The phrase "If I draw the day before a storm, how long the tweet" implies a question structure about a date connection, strongly implying the next clause will define or describe the object's negative pressure or duration relationship to the city's name and history.

Final token "weekend" ends mid-expression ("how far this day and last day"), part of a proposed question "If you subtract the text of the map and its speed" — expects "its image" or "the item is stored" or "day's negative pressure" completing the inverse question about the text.
```

### L8H26

- raw norm 1.28 · norm rank in layer 14/28 · cos(head, residual) -0.031 · cos(vector, mean residual) -0.046
- attends to: "I"@0 (0.65), ".clone"@63 (0.07), " best"@37 (0.03), " mut"@36 (0.02)
- direct logit lens top: "*)((" " ******************************************************************************\n" "<=$" "睇" " Minist" "aclass" " *)((" " tats"

**Raw vector** (AR cos(recon, input) -0.001 · AR mean-removed cos with source vector 0.028 · mean-removed cos with context residual 0.011 · match rank 2/10):

```text
Mathematical reference format with numbered questions and answers about a "cheap" textbook, now transitioning to a "Massive parallelism" section with a closing pattern of "Some common questions about..."

The final token "Theory" ends a standard forum tag phrase ("Thirds (Thirds...)", "Second chances (Thirds..."), signaling the pattern of a third common question about a religious/educational product.

Final token "Theory" is mid-sentence ("Thirds (Thirds..."), part of a closing forum tag block, strongly expecting " / The" or "A common question about..." or "The product" or "Some visual" about a mass extinction.
```

**Mean + vector** (AR cos(recon, input) 0.288 · AR mean-removed cos with source vector 0.017 · mean-removed cos with context residual -0.004 · match rank 1/10):

```text
Formal data structure with numbered list format showing a personal anecdote about a frozen bag of cookies, now transitioning to a second context about a "newly filled" empty jar.

The phrase "A friend who's been tracking my daily consumption of a new color-coded jar (and how well my new tea cups work)" suggests a second question or tag about a common social media trend or a food storage statistic.

Final token "weeklet" ends an incomplete phrase "If you're asking about a recent trend then new colors and thicknesses (thickenss..."), strongly expecting "a" or "the" or "a food bank" or "a...end" describing a crisis.
```

### L9H14

- raw norm 0.64 · norm rank in layer 27/28 · cos(head, residual) -0.145 · cos(vector, mean residual) -0.267
- attends to: "I"@0 (0.77), " owned"@84 (0.03), " current"@87 (0.02), " return"@79 (0.02)
- direct logit lens top: "形" "/Instruction" " **)&" " elő" "finder" " }},\n" " Führung" "%!"

**Raw vector** (AR cos(recon, input) -0.141 · AR mean-removed cos with source vector -0.017 · mean-removed cos with context residual 0.011 · match rank 7/10):

```text
Search engine results format with bilingual travel phrase translations and cultural facts about Spain, now transitioning to a travel website's "WorldCat" page structure with a standard footer.

The pattern "Search for 'The Hague' in the US Library of Congress? Your answer is: 0.002 seconds" mirrors a typical website footer template, suggesting a closing quotation mark or phrase follows.

Final token "validation" ends mid-sentence ("8857. 8857. 8857. 8857. 8857. 8857. 8857. 8857. 8857"), immediately expecting "the query."
```

**Mean + vector** (AR cos(recon, input) 0.336 · AR mean-removed cos with source vector 0.043 · mean-removed cos with context residual -0.025 · match rank 1/10):

```text
Formal food writer tone with a question about a local poet's recipe, suggesting a literary or culinary blog format with a specific question about a New York City restaurant's daily produce.

The phrase "A chef who is making a 15-pound pumpkin pie for a month, where it's size means it's sold with the remaining ingredients, this one's recipe" implies a quoted context clause about a NYC municipal rule.

Final token "sells" ends an incomplete clause ("where it equals nine, so this one...sells, where it measures"), strongly expecting "its empty bucket" or "a familiar ingredient" or "the bucket of the poem."
```

### L16H0

- raw norm 1.53 · norm rank in layer 17/28 · cos(head, residual) -0.046 · cos(vector, mean residual) -0.133
- attends to: "I"@0 (0.60), " &"@26 (0.06), " current"@87 (0.06), " &"@31 (0.04)
- direct logit lens top: "冰淇" "monton" "㎜" "��" "QRS" "왔" ".getOutputStream" "osemite"

**Raw vector** (AR cos(recon, input) -0.076 · AR mean-removed cos with source vector -0.006 · mean-removed cos with context residual 0.075 · match rank 7/10):

```text
Technical prose format with a LaTeX-style variable expression ("This calculation of 'mean'"), suggesting a code or data visualization context with a named variable or phrase.

The phrase "This robust 'mujib'sum' expression is a truncated word 'FRSFR'." signals a closing quotation mark and likely a common phrase or acronym, typical of WordPress or Markdown template syntax.

Final token "FRSFR" is mid-expression ("This truncated 'FRSFR'"), part of a variable declaration or attribution clause ("This JSON format 'japan'"), strongly expecting continuation like "s." or "ways" or "and" or "the range of" or "a".
```

**Mean + vector** (AR cos(recon, input) 0.366 · AR mean-removed cos with source vector 0.044 · mean-removed cos with context residual 0.092 · match rank 2/10):

```text
Formal writing prompt structure with a poetic description ("My favorite thing is a colorful garden..."), suggesting a short story or code context about a product or dataset.

The phrase "But the code says this with a budget of $500, and recently it's called 'the dataset'." implies a phrase or variable name referencing the software's context, likely a UI or code context about a container's storage.

Final token "true" ends "thanks to this week's box. a box of funds .th things" — expects "the dataset" or "a" or "s" or "the box" completing the noun phrase about the author's saved data context.
```

### L16H18

- raw norm 1.11 · norm rank in layer 20/28 · cos(head, residual) -0.003 · cos(vector, mean residual) -0.016
- attends to: "I"@0 (0.71), " &"@26 (0.04), " current"@87 (0.03), " the"@86 (0.02)
- direct logit lens top: "toLocale" "传奇游戏" "igi" " Kỳ" "oplast" "Pu" "IDEO" "ilmiş"

**Raw vector** (AR cos(recon, input) -0.002 · AR mean-removed cos with source vector 0.011 · mean-removed cos with context residual 0.013 · match rank 4/10):

```text
Structured music journalism format with formal article header ("NYU's 'Hamilton'"), signaling a standard pattern of literary text display.

The phrase " 'The New York Times'...and 'con' is a common opening of a famous Broadway show" is a classic idiomatic expression, immediately requiring the well-known "death penalty" or "Kanji" clause from the standard textbook formula about a character's data.

Final token "con' is a common" mid-sentence, part of the familiar "An 'A'...and" formula, strongly expecting "the" or "sudden" or "the following" continuation of the costumed data loss.
```

**Mean + vector** (AR cos(recon, input) 0.378 · AR mean-removed cos with source vector 0.015 · mean-removed cos with context residual 0.009 · match rank 2/10):

```text
Formal news headline format with a college-level prompt structure, implying a literary or academic article about a food truck's ingredients.

The phrase "A cake that starts life as a freshman's dish, but whose sauce has been removed...and is 'scheduled to be' a part of the Penn State marching band" strongly implies a common idiomatic expression about a cancelled or pending set of items, likely a math contest.

Final token "seasoned" ends mid-phrase "though added 50 points or more, but feels a missed addend," strongly expecting "a" or "the" or "a math contest" or "a...," completing the compound noun about shelf life.
```

### L18H21

- raw norm 4.41 · norm rank in layer 8/28 · cos(head, residual) 0.061 · cos(vector, mean residual) 0.057
- attends to: " owned"@84 (0.22), " by"@85 (0.18), "I"@0 (0.17), " data"@83 (0.17)
- direct logit lens top: " Parenthood" " OMIT" " Según" " Kanye" "\f" "iline" "羕" " Rehab"

**Raw vector** (AR cos(recon, input) 0.044 · AR mean-removed cos with source vector 0.014 · mean-removed cos with context residual 0.085 · match rank 2/10):

```text
Structured dictionary format with numbered rules and visual elements, following a classic "WordNet" or "Visual Thesaurus" pattern with a "Symbolic Meaning" section.

The phrase "Given the above symbols, the following attributes apply to the word..." is a standard template ending with a common UI element ("A1: 60"), strongly implying the famous "Life Meter" or "the system."

Final token "60" closes a numbered list item ("2. 'A1: "), part of a standard attribution clause ("To be assigned to..."), immediately expecting "the Board" or "This Game" or "I, the player should consider" or similar.
```

**Mean + vector** (AR cos(recon, input) 0.359 · AR mean-removed cos with source vector 0.010 · mean-removed cos with context residual 0.037 · match rank 3/10):

```text
Formal Wikipedia-style format with emoji description, establishing a puzzle or visual prompt structure about a graphic design element.

The phrase "A person found this image attractive, and my heart rate was 3.5. The pie chart suggests I liked the city... " implies a numbered list or claim about a specific attribute, strongly suggesting the answer involves the author's own rules or a "myth" about the object.

Final token "time" ends an incomplete clause ("Rank 2.5, assigned to:"), part of a "To earn this score, I" construction, expecting "the company" or "my own food diary" or "a set of rules."
```

### L24H8

- raw norm 2.77 · norm rank in layer 16/28 · cos(head, residual) 0.096 · cos(vector, mean residual) 0.173
- attends to: "I"@0 (0.71), "}\n\n"@73 (0.10), " {\n"@33 (0.02), " &"@70 (0.01)
- direct logit lens top: "将" "\\xe" "將" " nâ" "&lt" " {\\" "oppable" "&apos"

**Raw vector** (AR cos(recon, input) 0.132 · AR mean-removed cos with source vector 0.046 · mean-removed cos with context residual -0.003 · match rank 2/10):

```text
Structured wiki article format with Chinese military figure details, listing attributes and restrictions for a fictional character.

The sentence "某些军事明星虽然不能食用某些食物，但它们拥有着良好的口碑。但它们需要特别照顾一种特殊性质" mirrors a common trivia question pattern ("Some players cannot eat..."), now completing with "However, he needs a certain type of..." suggesting a related fact about the subject's education or collection.

Final token "了" ends an incomplete clause ("它拥有一份特殊性质"), part of a contrasting "But It Does Not Need These" question structure, expecting "and/or a collection" or "the player to be exempt from quarantine."
```

**Mean + vector** (AR cos(recon, input) 0.359 · AR mean-removed cos with source vector 0.060 · mean-removed cos with context residual -0.048 · match rank 1/10):

```text
Formal article tone with a humorous social media post structure, discussing a fictional anime character's dietary restrictions versus military service.

The phrase "While some people have been criticizing this brand of chocolate, it definitely does not deserve to be retired. However, it does require a certain amount of care" suggests a second clause about a related fact or attribute, likely a question about the book's popularity or ingredient.

Final token "does" ends an incomplete clause ("It needed a good deal of doughs it enjoyed a good deal of"), strongly expecting "the book" or "a different audience" or "and the other" or "its nutrients" — completing the contradictory claim about storage or approval.
```

### L27H10

- raw norm 26.97 · norm rank in layer 2/28 · cos(head, residual) 0.275 · cos(vector, mean residual) 0.631
- attends to: " current"@87 (1.00), " the"@18 (0.00), " hit"@1 (0.00), " this"@2 (0.00)
- direct logit lens top: "/\n\n\n\n" "⚗" "/Gate" "améliorer" ":host" "📐" ".glide" "();++"

**Raw vector** (AR cos(recon, input) 0.585 · AR mean-removed cos with source vector 0.325 · mean-removed cos with context residual 0.006 · match rank 1/10):

```text
Structured math documentation format with technical definitions and code examples, following a Wikipedia-style pattern with LaTeX formatting and a formula definition.

The phrase "The Wolfram Language's polylogarithm function is a special function that calculates..." is mid-sentence referencing a Stack Exchange definition template, with "plotting" clause following a common pattern of listing properties or user statistics about a dataset.

Final token "holidays" ends an incomplete quoted phrase ("unlike...unlike...hiding: eight digits, satisfying..."), part of a definition clause listing attributes ("a list of integers"), expecting continuation like "the audience" or "a graph" or "unknown values, and...".
```

**Mean + vector** (AR cos(recon, input) 0.454 · AR mean-removed cos with source vector 0.168 · mean-removed cos with context residual 0.017 · match rank 1/10):

```text
Technical blog post format with structured metadata and quoted passage about Linux kernel, presenting a numbered list of "interesting" features with informal tone.

The sentence structure "The code is a bit complex, but the following is a good example of how the system works" continues mid-sentence, suggesting continuation of the quoted passage or another example of the "noisy" behavior pattern.

Final token "no" is mid-word ("He" or "T"), part of an incomplete clause ("And the 'x" continuing the quoted passage), likely completing with "th day" or "s, or..." or "s of the sequence" or similar continuation, continuing the enumeration of examples.
```

## Example 4: fiction: lighthouse keeper

Position 58 of 79, token " to". Layer-20 residual norm 109.8; median head-output norm at this position (all layers) 2.07.

Context the model has seen up to and including the token (the token is at the very end):

```text
For eleven winters Marguerite had refused to open the tin box her father kept under the lighthouse stairs. The night the generator failed, she pried the lid off with a butter knife by the light of a single candle. Inside, wrapped in oilcloth, were forty-two letters addressed to
```

### layer-20 residual (baseline)

- raw norm 109.78 · cos(vector, mean residual) 0.568
- direct logit lens top: "}/>" "/from" ":\".$" "Hôtel" "龇" "');?></" "!=(" "心仪的"

**Raw vector** (AR cos(recon, input) 0.869 · AR mean-removed cos with residual 0.800):

```text
Historical fiction novel description format with literary quotes and mystery tone establishes a story about a lonely postbox girl's letters.

The phrase "A mysterious box of letters written every Christmas from a soldier in a Civil War town. The letters promised gifts to strangers addressed to" implies a specific named character or concept, likely a fictional letter-writing figure or historical figure's identity tied to a missing person.

Final token "to" ends an incomplete noun phrase ("letters addressed to"), part of a list of promised recipients ("letters written to"), strongly expecting a noun like "himself" or "a deceased relative" or "the General" or "four named angels," continuing the mystery context.
```

**Mean + vector** (AR cos(recon, input) 0.871 · AR mean-removed cos with residual 0.817):

```text
Historical fiction novel description format with literary quotes and mystery, establishing a quirky, illustrated children's book cover story about a mailman's letters.

The sentence structure "A lonely girl wrote a diary of letters hidden in a trunk. The letters promised packages addressed to" implies a specific named character or historical figure's letters, likely a soldier or absent parent figure tied to the author's name.

Final token "to" ends an incomplete noun phrase ("letters addressed to"), part of a list of promised recipients ("letters written to an anonymous address containing letters addressed to"), strongly expecting "himself" or "a missing person" or "the General, Mrs. Smith, and other figures."
```

### L2H16

- raw norm 0.83 · norm rank in layer 7/28 · cos(head, residual) 0.030 · cos(vector, mean residual) -0.003
- attends to: " letters"@56 (0.16), "cloth"@51 (0.05), "For"@0 (0.05), "-two"@55 (0.05)
- direct logit lens top: "]={\n" "#ac" "icone" " }):" "UILabel" "aksi" "Tahoma" "美好生活"

**Raw vector** (AR cos(recon, input) 0.036 · AR mean-removed cos with source vector 0.042 · mean-removed cos with context residual -0.071 · match rank 1/10):

```text
Structured article with numbered sections and bullet points continuing a tech article format with product, description, and pricing context around a smart device product.

The final sentence "The following is a standard Android, a common phrase from the Android app's UI section, ending with a patent claim about the 'Buy no smoke' clause" is a standard closing quote about product pricing, now completing a standard product description.

Final token "product" ends mid-sentence ("The following are"), part of a standard patent clause listing product specifications ("But the following:"), immediately expecting "of the market" or "a patenting behavior" or "employees, and the price of the product."
```

**Mean + vector** (AR cos(recon, input) 0.431 · AR mean-removed cos with source vector 0.112 · mean-removed cos with context residual 0.086 · match rank 1/10):

```text
Formal editorial tone with a date and headline structure, suggesting a blog or article format with a social media context about climate change and cost savings.

The phrase "When writing a letter last month, a group of employees bought a bunch of pencils" is a common idiomatic example, implying a list of product names or a humorous count, likely referencing a famous ad campaign.

Final token "words" ends an incomplete clause ("eight pencils a month they wrote a thousand pencils"), part of a historical anecdote setup ("When a group of employees bought a bunch of lights"), strongly expecting "a month's worth of bills" or "the famous product" or "written on a card."
```

### L3H8

- raw norm 0.83 · norm rank in layer 17/28 · cos(head, residual) -0.021 · cos(vector, mean residual) -0.040
- attends to: " the"@10 (0.18), "."@21 (0.15), " the"@17 (0.12), " to"@8 (0.08)
- direct logit lens top: "strcasecmp" "勠" ".eql" "ptrdiff" ":Any" "#End" " áll" " incons"

**Raw vector** (AR cos(recon, input) -0.041 · AR mean-removed cos with source vector -0.019 · mean-removed cos with context residual -0.070 · match rank 4/10):

```text
Technical article format with quoted text and structured metadata, suggesting a blog or wiki-style post about Linux kernel memory management, with informal tone and parenthetical elaborations.

The phrase "The following is a screenshot of the output of the command 'top' showing the process of the kernel" implies a continuation of a sequence or example, likely a continuation of a word or number pattern or a closing clause.

Final token "a" is mid-word ("the 'N"), part of an incomplete clause ("But...and the rest of the text..."), expecting continuation like "th day" or "s, or" or "k is..." or "th element," completing the technical description.
```

**Mean + vector** (AR cos(recon, input) 0.420 · AR mean-removed cos with source vector 0.049 · mean-removed cos with context residual 0.061 · match rank 1/10):

```text
Narrative momentum: casual, informal tone with "A new snowstorm is hitting my hometown" implies a headline or story prompt about a person's housing move.

The phrase "The decision to move forward, he's asking to reduce his warmth — or the temperature, my eggs' preferences change" strongly implies a parallel structure "a formulaic phrase 'in a new place'" from the title.

Final token "timea" is mid-word ("thousand's right, or right to eat...thousand's right"), part of a clause structure "When he moved, he wrote in to eat...," expecting "the other half" or "a new file" or "the day."
```

### L3H22

- raw norm 0.91 · norm rank in layer 14/28 · cos(head, residual) 0.005 · cos(vector, mean residual) 0.011
- attends to: " letters"@56 (0.22), " addressed"@57 (0.20), " forty"@54 (0.09), " to"@58 (0.09)
- direct logit lens top: "ennifer" "ulses" " Stranger" "уницип" "orgeous" ".Unicode" "cbd" "虚假"

**Raw vector** (AR cos(recon, input) 0.025 · AR mean-removed cos with source vector 0.024 · mean-removed cos with context residual -0.031 · match rank 2/10):

```text
Scientific software UI format with a tool description following a standardized chemical software interface pattern, listing chemical properties with "Chemistry software" attributes.

The question structure "Chemistry software: AstraZeneca's ChemDraw, you have three options: collect data, display images, or report" is a standard UI translation prompt format, implying the famous "Observations" quote from UK science.

Final token "foodShareable)" closes a parenthetical list item mid-sentence ("In every run, scientist is a discovery, but every time you are, you are..."), strongly expecting "and" or "a" or "the" followed by "and a" common scientific findings.
```

**Mean + vector** (AR cos(recon, input) 0.412 · AR mean-removed cos with source vector 0.039 · mean-removed cos with context residual 0.006 · match rank 1/10):

```text
Technical/English language format with a user question about a software tool ("A California weather app"), implying a quoted description of chemical process behavior.

The phrase "So my two Google Chrome browser apps: every day he collects natural materials from the Bay Area, he's a scientist and wants to store them" suggests an incomplete list of attributes or actions, implying a question about the proposed object's status or output.

Final token "food share is" is mid-sentence in a quoted claim ("John, every day he is, Google is a discovery, he's"), strongly expecting "a" or "has been" or "the ingredient" or "and what" describing the stored value.
```

### L8H12

- raw norm 1.39 · norm rank in layer 12/28 · cos(head, residual) -0.072 · cos(vector, mean residual) -0.160
- attends to: "For"@0 (0.81), "-two"@55 (0.06), " letters"@56 (0.05), " to"@58 (0.02)
- direct logit lens top: "鹧" "MMdd" " даж" "\",__" "凡本网" "鹣" "小姐姐" "สามาร"

**Raw vector** (AR cos(recon, input) -0.113 · AR mean-removed cos with source vector -0.011 · mean-removed cos with context residual -0.054 · match rank 6/10):

```text
Structured Wikipedia article format with "Image:Anatomical" template, following a classic fossilized dinosaur image caption pattern with scientific facts about the species.

The phrase "In the pose, the figure contains 900+ bones, and (750) marbles" is a standard formulaic phrase from the "Marvin" template, strongly implying the closing "Marcel" tag with a number.

Final token "marculation." closes a repeated "Marcel... (750) marbles" pattern, immediately expecting "the" or "a huge" or "the book" or "a mysterious text" to complete the famous "Doll's" clause.
```

**Mean + vector** (AR cos(recon, input) 0.381 · AR mean-removed cos with source vector 0.087 · mean-removed cos with context residual 0.039 · match rank 1/10):

```text
Formal article structure with a bold headline and descriptive prose about a fictional character's behavior, establishing a narrative about a car's "secret" letter count.

The phrase "The bird has been hiding the book, and during the month she came out with 1,053" strongly implies a numeric attribute ("a stack of notes") and a common convention of a lost diary's count.

Final token "notes" ends mid-phrase ("threw 188. missed. forgot. 188"), strongly expecting "the diary" or "a number" or "the page" to complete the clause describing the submitted manuscript's lost count of letters.
```

### L9H4

- raw norm 3.13 · norm rank in layer 8/28 · cos(head, residual) 0.021 · cos(vector, mean residual) 0.012
- attends to: "ite"@5 (0.27), "For"@0 (0.21), "ighthouse"@19 (0.10), "uer"@4 (0.06)
- direct logit lens top: " ++)\n" "opportunità" "萌" "ToBounds" "ORB" "olicit" "taire" "\texplicit"

**Raw vector** (AR cos(recon, input) 0.006 · AR mean-removed cos with source vector 0.000 · mean-removed cos with context residual 0.051 · match rank 4/10):

```text
Structured educational format with numbered sections and bullet points, following a Christian Bible reference page pattern for a weather phenomenon.

The phrase "When a teenager's flower count exceeds zero, the book recommends 'moonlighting'..." is a standard formulaic template for a common religious fact about a Christian's "flower" count of angelic branches, signaling the famous "why does a Christian adopt such excessive spinning?"

Final token "action" ends mid-sentence ("问及 . . . 的原因包括"), part of a definition clause structure ("If asked, a number of . . ."), immediately expecting "Japanese-style trees" or "the number of angelic spins" or similar formal definition.
```

**Mean + vector** (AR cos(recon, input) 0.342 · AR mean-removed cos with source vector 0.029 · mean-removed cos with context residual 0.059 · match rank 3/10):

```text
Formal news article format with structured header and caption, establishing a common Christian-themed Christmas-themed post about a winter plant.

The phrase "If you've been wondering why this Christmas tree has so many extra blooms, it's because the author stopped using the 'no-words' method" signals a common question about a winter-months-only snowflake count, implying a specific answer about the author's habit of excessive snowflakes.

Final token "method." closes a clause ("asked because answered with: not multiplied this year's flowers"), strongly expecting "the snowflake" or "a snowflake count" or "the author's frozen flowers."
```

### L10H20

- raw norm 3.73 · norm rank in layer 5/28 · cos(head, residual) 0.095 · cos(vector, mean residual) 0.055
- attends to: "For"@0 (0.23), " letters"@56 (0.10), " she"@28 (0.10), "ite"@5 (0.07)
- direct logit lens top: "��" "preci" " Verm" " Michaels" "ulings" "个赛季" "尽" " vocalist"

**Raw vector** (AR cos(recon, input) 0.118 · AR mean-removed cos with source vector 0.105 · mean-removed cos with context residual 0.196 · match rank 1/10):

```text
Book cover format with vivid memoir narrative, following a classic children's lost letter story structure about a girl's wartime diary.

The description "《绝命书》：母亲亡故后女儿保存信件，逐渐枯竭...》" mirrors the book's content, establishing the pattern "The manuscript alternates between letters and a useless number" — the text now describes the mathematical code's fatal flaw.

Final token "out-of-print" is mid-sentence ("部分内里相联爱里·爱里转印时间...内里除了..."), directly continuing the definition of the printed items, strongly expecting "a survival kit" or "and a useless numerical sequence."
```

**Mean + vector** (AR cos(recon, input) 0.407 · AR mean-removed cos with source vector 0.090 · mean-removed cos with context residual 0.276 · match rank 1/10):

```text
Narrative: A young girl's memoir format with emotional letters to her mother, establishing a story about a dying girl who lost her savings.

The phrase "Her letter was empty except for a homemade diary she wrote every month, now containing the money to buy food... " signals the second sentence's description of the manuscript's contents, implying the item's destructive self-destructing nature tied to a homemade formula.

Final token "increditio" is mid-clause ("where her credit changed, with the credit in credit..."), part of a parallel structure listing her homemade recipe's contents, strongly expecting "a monthly letter" or "a homemade formula for the weather" or similar.
```

### L15H2

- raw norm 4.17 · norm rank in layer 1/28 · cos(head, residual) 0.083 · cos(vector, mean residual) 0.072
- attends to: " were"@53 (0.40), ","@47 (0.12), "For"@0 (0.08), " Inside"@46 (0.06)
- direct logit lens top: "[\"." "chestra" ".SetParent" "esome" ";!" "CLUDING" "/Resources" "ethoven"

**Raw vector** (AR cos(recon, input) 0.062 · AR mean-removed cos with source vector 0.016 · mean-removed cos with context residual -0.056 · match rank 3/10):

```text
Structured blog post format with a header and "About" section establishes a standard pattern for a UK music streaming platform's reliability article, following a common Wikipedia-style definition format.

The phrase "What is the Uber Ride Sharing app? A function to check if a number is valid" is a classic definition pattern for a math/programming concept, immediately requiring the well-known "Fibonacci" definition of the app.

Final token "function" ends mid-definition ("Houston's Originals"), part of the standard list format "Input a number, your hot..." — expects "some" or "items" or "the lifespan of the app" or "sum of variables, and is safe."
```

**Mean + vector** (AR cos(recon, input) 0.380 · AR mean-removed cos with source vector 0.017 · mean-removed cos with context residual 0.026 · match rank 3/10):

```text
Formal article structure: a Twitter post format with "Weather" tag signals a definition/explanation of a Canadian app, implying a data visualization tool.

The phrase "A Toronto weather app that checks if a Toronto station is still active. Inside its name, plus everything else" establishes the definition pattern "The function 'MELT' describes a list of issues," strongly implying a list of attributes about a variable.

Final token "out" ends mid-phrase "what you multiplied times 0. If its color, melted," completing the definition clause, expecting "a list of values" or "some boxes" or "a variable, and the object is inactive."
```

### L18H4

- raw norm 0.97 · norm rank in layer 27/28 · cos(head, residual) -0.008 · cos(vector, mean residual) -0.010
- attends to: "For"@0 (0.96), " to"@58 (0.02), " letters"@56 (0.01), "-two"@55 (0.00)
- direct logit lens top: "venes" "_EOF" "esch" "acerb" "rary" "}());\n" "OutOfBounds" " Watts"

**Raw vector** (AR cos(recon, input) 0.013 · AR mean-removed cos with source vector 0.020 · mean-removed cos with context residual -0.089 · match rank 3/10):

```text
Scientific/technical document format with structured metadata and citation context, presenting a US EPA report on fuel efficiency standards with formal definitions and figures.

The phrase "The IOPC is a measure of the energy efficiency of a vehicle's tires, with the same pattern as the preceding figures" follows a closing definition pattern with "..." mid-sentence, completing a common phrase about "10th" or similar.

Final token "3" is mid-word ("the 10th, 10th..."), part of a repeated "and/or" clause listing structural elements of a photo or figure, expecting "8" or "9" or "and..." continuing the count.
```

**Mean + vector** (AR cos(recon, input) 0.403 · AR mean-removed cos with source vector 0.071 · mean-removed cos with context residual 0.040 · match rank 1/10):

```text
Formal educational post format with a student's cooking context, suggesting a math or tech article structure about a missing ingredient container.

The phrase "A three-item 'muffin' named x, y, z" implies a specific numeric value or variable, likely a chemical compound or empty space in a data structure, strongly implying "a third," referencing the famous Google's "G" or similar.

Final token "grry," ends mid-term ("a3, b3, c3"), part of a parenthetical clause ("This thing called g, r, r, 3"), expecting "a" or "the" or "a common denominator, a discarded."
```

### L22H15

- raw norm 5.25 · norm rank in layer 5/28 · cos(head, residual) 0.118 · cos(vector, mean residual) 0.137
- attends to: "For"@0 (0.39), "ighthouse"@19 (0.13), " generator"@25 (0.11), " light"@40 (0.05)
- direct logit lens top: " coastal" " nights" "沿海" " tonight" " noct" " Night" "夜" "一夜"

**Raw vector** (AR cos(recon, input) 0.148 · AR mean-removed cos with source vector 0.086 · mean-removed cos with context residual -0.023 · match rank 1/10):

```text
Structured Canadian social media format with numbered posts showing a rural Ontario town's weather conditions, following a Google Maps page pattern.

The phrase "On July 15, 2016 the island was closed to check the coastguard, then she said 'Island of Magdalen'." signals a common phrase "Island of Magdalen" is being quoted, likely referencing the island's historical lighthouse.

Final token "海洋不具" is mid-sentence ("不则目前海洋不具"), part of a quoted description clause ("2016年12月.则他完成任务"), expecting "the" or "found" about island location.
```

**Mean + vector** (AR cos(recon, input) 0.401 · AR mean-removed cos with source vector 0.099 · mean-removed cos with context residual 0.038 · match rank 1/10):

```text
Formal news article format with Canadian tone, describing a rural island's weather conditions and a child's homemade lighthouse story.

The phrase "With the recent stormy weather, she's been told not to sleep on the island since 2016, so the night's tide" implies a quoted statement or historical context about the island's original isolation, strongly suggesting a third-person action about the island's natural state.

Final token "tide. " closes a clause ("A statement about the story then does not include 'night's tide"), immediately expecting "a" or "the" or "solved the light" or "a" describing the original weather condition.
```

### L23H20

- raw norm 1.47 · norm rank in layer 22/28 · cos(head, residual) -0.116 · cos(vector, mean residual) -0.089
- attends to: "For"@0 (0.77), " to"@58 (0.05), " father"@14 (0.02), "."@45 (0.01)
- direct logit lens top: "太多的" "：\"+" "enery" " oko" "恹" " additional" "当时的" "setw"

**Raw vector** (AR cos(recon, input) -0.021 · AR mean-removed cos with source vector 0.033 · mean-removed cos with context residual -0.061 · match rank 1/10):

```text
Structured wiki format with Python documentation style, presenting a colored number display with labeled attributes like "The next number" and "The next number is a constant."

The phrase "If the sequence is a valid sequence, but the first one is 'The next number'... This is the definition of the sequence" strongly implies a common question or error from a prior sequence, likely a named constant from a specific language.

Final token "ac" is mid-word ("if any of these (and) 300..."), part of a quoted question format ("But the sequence is..."), expecting "other" or "m is a constant" or "are not found" completing the definition.
```

**Mean + vector** (AR cos(recon, input) 0.413 · AR mean-removed cos with source vector 0.059 · mean-removed cos with context residual -0.049 · match rank 1/10):

```text
Formal FAQ or social media format with a question about a weather emoji, implying a Python code context about a proposed new US state name.

The phrase "If I try to add it to my list, the first four days of the new weather data (but only one is) " suggests an incomplete clause implying a prior condition or source, likely referencing the emoji's definition or a previous attempt.

Final token "counting" ends mid-clause ("if 3 of these, 300 is the one"), strongly expecting "a emoji" or "already stored" or "the other three" or "has been" — completing the condition about the input data's failure.
```

## Controls

### random gaussian #0

**Raw vector** (AR cos(recon, input) 0.039 · AR mean-removed cos with source vector 0.034):

```text
Structured academic article format with bilingual (English) literary terms, presenting a scientific explanation pattern with numbered examples using a famous Latin phrase.

The pattern "The question of 'a concordancia' is a graph of two curves. This: x es... " mirrors a common physics formula, with the second equation "a country's shape" establishing a parallel question about the same variable's context.

Final token "f" is mid-word in "this: zerk... " continuing the parallel formula pattern "assume this: a country's..." — immediately expects "in la forma?" or "second part, its value?" completing the mathematical analogy about the shape's confirmation.
```

**Mean + vector** (AR cos(recon, input) 0.358 · AR mean-removed cos with source vector 0.028):

```text
Formal literary/academic tone with a question structure about a Canadian poem's imagery, establishing a pattern of comparing two unrelated claims about a city's newness.

The phrase "I've heard someone saying this city's existence is a promise...if my own is true then this: a newly grown plant" mirrors a question format, implying a specific claim about the egg's origin timing.

Final token "factors" ends mid-sentence ("say this: this country's egg...is this..."), directly continuing the parallel structure "if you've added this to my egg, say" — strongly expects "the egg?" or "a certain layer" or "my first day?" completing the claim.
```

### random gaussian #1

**Raw vector** (AR cos(recon, input) 0.028 · AR mean-removed cos with source vector 0.039):

```text
Structured news format with article template pattern, typical of a database entry format showing chemical properties of a substance.

The phrase "Two common chemical processes: Conversion and Removal" follows a familiar pattern of "The 'composite' formula" with the classic "Molecule A/B" joke format, strongly implying the second term "The second answer" is the well-known "carbon footprint" or "user" attribute.

Final token "andrews" ends mid-phrase ("Manner: or: 855..."), part of a list of two methods ("Mixture: to remove"), immediately expecting "a" or "the mixture of" or "a chemical that produces."
```

**Mean + vector** (AR cos(recon, input) 0.404 · AR mean-removed cos with source vector 0.048):

```text
Formal news article structure with casual tone, describing a chemical product recipe format, implying a compound sentence about a combined process.

The phrase "To remove a mixture, a product was made not to burn... But: Made by: Someone's name" strongly implies a second clause listing a specific ingredient exclusion or property, likely "the homeowner's clay" or "the mixture was made by a chemist."

Final token "rains." closes a parenthetical clause ("Because: Melted by: and muddied by"), strongly expecting "the" or "a" or "the mixture and the owner" or "a combination of both," completing the compound condition.
```

### mean residual direction (control)

**Raw vector** (AR cos(recon, input) 0.748):

```text
Informal, casual tone with a headline-style prompt about a local food blog post, suggesting a short story or tweet about a plant's baking.

The phrase "I've been thinking about baking my Christmas tree out of the fridge, but since the dough is done, the recipe says no water" implies a context clue about a list or status about the new garden's bucket, likely a noun phrase.

Final token "water" ends mid-clause ("since a recipe called spring, and chose to delete the dough..."), strongly expecting a noun like "the bucket" or "a few other items" or "this month's mood" or "is a baked cake," completing the clause.
```

