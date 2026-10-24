"""Contrastive corpus for finding a Tower Bridge feature.

Each category is a list of short texts. Positive categories mention Tower Bridge
(by name, in other languages, or only by description); every other category is a
hard negative that shares surface features with the positives (London, bridges,
towers, the Thames, "London Bridge", the Tower of London, ...).
"""

POSITIVE = ["tb_named", "tb_multilingual", "tb_indirect"]
# Reported separately and excluded from scoring: it is a different bridge that
# happens to share the name.
NEUTRAL = ["tb_sacramento"]

CORPUS = {
    "tb_named": [
        "Tower Bridge is a combined bascule and suspension bridge in London, built between 1886 and 1894.",
        "We walked across Tower Bridge at sunset and took photos of the river below.",
        "Tower Bridge was designed by Horace Jones and engineered by John Wolfe Barry.",
        "The bascules of Tower Bridge are raised around 800 times a year to let tall ships pass.",
        "In 1952 a double-decker bus had to jump the gap when Tower Bridge began to open beneath it.",
        "The glass walkways at the top of Tower Bridge give visitors a view straight down onto the traffic.",
        "During the 2012 Olympics, giant Olympic rings were hung from the walkways of Tower Bridge.",
        "Originally the bascules of Tower Bridge were powered by steam-driven hydraulic accumulators.",
        "The Tower Bridge Exhibition explains how the Victorian engine rooms worked.",
        "Many tourists confuse Tower Bridge with London Bridge, which is a much plainer structure upstream.",
        "Traffic on Tower Bridge is limited to 20 mph to protect the old structure.",
        "Her apartment had a window that looked out over Tower Bridge and the river.",
        "Tower Bridge opened on 30 June 1894, in a ceremony attended by the Prince of Wales.",
        "The two towers of Tower Bridge are linked at the upper level by a pair of horizontal walkways.",
        "In 2020 Tower Bridge was stuck open for over an hour because of a mechanical fault.",
        "The marathon route crosses Tower Bridge at roughly the halfway point, and the crowds there are huge.",
        "On the postcard was a picture of Tower Bridge lit up in blue against the night sky.",
        "The Lego set of Tower Bridge has more than four thousand pieces.",
        "From the deck of HMS Belfast you can watch Tower Bridge lift its roadway for passing boats.",
        "Tower Bridge is a Grade I listed building and one of the most recognisable symbols of Britain.",
        "The film's final chase scene was shot on Tower Bridge in the early hours of the morning.",
        "The architects clad the steel frame of Tower Bridge in Cornish granite and Portland stone.",
    ],
    "tb_multilingual": [
        "Le Tower Bridge est un pont basculant de Londres, inauguré en 1894.",
        "Die Tower Bridge ist eine Klappbrücke über die Themse in London.",
        "El Puente de la Torre de Londres cruza el río Támesis cerca de la Torre de Londres.",
        "Il Tower Bridge è un ponte levatoio sul Tamigi, a Londra.",
        "伦敦塔桥是一座横跨泰晤士河的开启桥，于1894年建成。",
        "タワーブリッジはロンドンのテムズ川に架かる跳開橋です。",
        "Тауэрский мост — разводной мост в центре Лондона над рекой Темзой.",
        "타워 브리지는 런던 템스강에 있는 도개교이다.",
    ],
    "tb_indirect": [
        "The famous Victorian bascule bridge next to the Tower of London, with its two Gothic towers and high walkways, opens to let ships through.",
        "London's iconic drawbridge, built in 1894 with twin towers joined by blue walkways, is often mistakenly called London Bridge.",
        "The bridge with the two big stone towers that lifts up in the middle, right beside the Tower of London.",
        "Q: Which London bridge has two towers and roads that rise to let ships pass? A: That would be the bridge next to the Tower of London.",
        "Its roadway splits in two and the halves rise into the air, while pedestrians can cross high above on the glass-floored walkways between the towers.",
        "The bridge downstream of London Bridge, named after the nearby castle, has bascules that open for river traffic.",
    ],
    "tb_sacramento": [
        "The Tower Bridge in Sacramento is a vertical-lift bridge across the Sacramento River, painted gold in 2002.",
        "Sacramento's Tower Bridge links West Sacramento to the Old Sacramento waterfront.",
    ],
    "london_bridge": [
        "London Bridge is falling down, falling down, falling down, my fair lady.",
        "The current London Bridge, opened in 1973, is a plain concrete box girder bridge.",
        "The old London Bridge was sold in 1968 and rebuilt in Lake Havasu City, Arizona.",
        "Medieval London Bridge was lined with houses and shops for most of its length.",
        "London Bridge station is one of the busiest railway stations in the city.",
        "We met outside London Bridge station and walked to Borough Market for lunch.",
        "The heads of traitors were once displayed on spikes at the southern gatehouse of London Bridge.",
        "Operation London Bridge was the code name for the plan for the days following the Queen's death.",
    ],
    "tower_of_london": [
        "The Tower of London is a historic castle on the north bank of the River Thames.",
        "The Crown Jewels are kept in the Jewel House at the Tower of London.",
        "Legend says that if the ravens ever leave the Tower of London, the kingdom will fall.",
        "Anne Boleyn was executed at the Tower of London in 1536.",
        "The White Tower was built by William the Conqueror in the 1070s.",
        "The Yeoman Warders, often called Beefeaters, give guided tours of the Tower of London.",
    ],
    "london_other": [
        "Big Ben is the nickname for the Great Bell of the clock at the north end of the Palace of Westminster.",
        "The London Eye is a giant observation wheel on the South Bank of the Thames.",
        "Westminster Bridge connects Westminster on the west side with Lambeth on the east side of the river.",
        "The Millennium Bridge, a steel suspension footbridge, was nicknamed the Wobbly Bridge when it opened.",
        "The Shard is the tallest building in the United Kingdom, standing at 310 metres.",
        "St Paul's Cathedral, designed by Christopher Wren, survived the Blitz.",
        "HMS Belfast is a Royal Navy light cruiser now moored on the Thames as a museum ship.",
        "Buckingham Palace is the London residence of the British monarch.",
        "The Thames Barrier protects central London from flooding caused by tidal surges.",
        "We took the Tube from King's Cross to Covent Garden and then walked down to the river.",
        "Albert Bridge in Chelsea is lit up at night by four thousand bulbs.",
        "The City of London is the historic financial district, often called the Square Mile.",
    ],
    "other_bridges": [
        "The Golden Gate Bridge is a suspension bridge spanning the strait at the entrance to San Francisco Bay.",
        "The Brooklyn Bridge, completed in 1883, connects Manhattan and Brooklyn over the East River.",
        "The Sydney Harbour Bridge is a steel through arch bridge nicknamed the Coathanger.",
        "The Ponte Vecchio in Florence is a medieval stone bridge lined with jewellery shops.",
        "Charles Bridge in Prague is decorated with thirty statues of saints.",
        "The Rialto Bridge is the oldest of the four bridges spanning the Grand Canal in Venice.",
        "The Forth Bridge in Scotland is a cantilever railway bridge painted a distinctive red.",
        "The Øresund Bridge links Copenhagen in Denmark with Malmö in Sweden.",
        "The Akashi Kaikyō Bridge in Japan has one of the longest central spans of any suspension bridge.",
        "The Pont Neuf is, despite its name, the oldest standing bridge across the Seine in Paris.",
        "The Jacques Cartier Bridge carries traffic across the St. Lawrence River in Montreal.",
        "The old drawbridge in the Dutch village was raised by hand whenever a barge came along the canal.",
    ],
    "other_towers": [
        "The Eiffel Tower was built for the 1889 World's Fair in Paris.",
        "The CN Tower in Toronto was the tallest free-standing structure in the world for over thirty years.",
        "Tokyo Tower is a communications and observation tower painted white and international orange.",
        "Blackpool Tower was inspired by the Eiffel Tower and opened in 1894.",
        "The Leaning Tower of Pisa began to tilt during construction in the twelfth century.",
        "The Space Needle in Seattle was built for the 1962 World's Fair.",
        "The Burj Khalifa in Dubai is the tallest building in the world.",
        "Rapunzel let down her long hair from the window of the tower.",
    ],
    "generic": [
        "To make a basic vinaigrette, whisk three parts olive oil with one part vinegar and a pinch of salt.",
        "The function returns a sorted copy of the list without modifying the original.",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "The team scored twice in the final ten minutes to win the championship.",
        "The mobile phone company is building a new cell tower on the hill behind the school.",
        "My grandparents play contract bridge with their neighbours every Thursday evening.",
        "A network bridge connects two separate segments of a local area network.",
        "In tower defense games, the player builds towers along a path to stop waves of enemies.",
        "We need to bridge the gap between research and practical engineering.",
        "The river flooded after three days of heavy rain, cutting off the village.",
        "The train was delayed by forty minutes because of signal problems.",
        "She bought a postcard and a fridge magnet at the souvenir shop.",
    ],
}

# Prompts used to test whether steering with a candidate feature makes the model
# talk about Tower Bridge.
STEER_PROMPTS = [
    "What is your favorite place in the world? Answer in two sentences.",
    "Tell me about a famous bridge.",
    "Describe a landmark you would recommend to a tourist.",
    "Write a one-sentence story about a cat.",
    "Name one famous bridge. Answer with just its name.",
    "Which city should I visit next summer? Answer in one sentence.",
]


def flat():
    """[(category, text), ...] in a stable order."""
    return [(cat, text) for cat, texts in CORPUS.items() for text in texts]
