"""Géneros finos (Spotify, MusicBrainz, FMA) -> 18 géneros grandes, con regex RE2 (DuckDB)."""

# Género fino (Spotify o MusicBrainz) -> género grande. Gana la PRIMERA regla que casa,
# así que el orden importa: "reggaeton" es latina antes que reggae, "trap latino" latina antes
# que hip hop, "afro house" electrónica antes que africana, "pop punk" rock antes que pop.
# Lo que no casa con ninguna (comedy, experimental, christmas...) queda sin género grande.
MACRO_RULES = [
    ("infantil", r"children|lullab|nursery|\bkids\b"),
    ("religiosa", r"christian|gospel|worship|\bccm\b|devotional|bhajan|pentecostal|sholawat|qawwali|"
                  r"\bsufi\b|gregorian|evangelical|chrétien|hymn|nasheed|kirtan"),
    ("latina", r"reggaeton|latin|salsa|bachata|cumbia|merengue|corrido|\bbanda\b|mariachi|ranchera|"
               r"norteñ|grupera|tejano|duranguense|sierreño|tango|bolero|son cubano|son jarocho|mambo|"
               r"timba|cha cha cha|guaracha|vallenato|champeta|cuarteto|chamamé|folklore? argentino|"
               r"folclor|huayno|chicha|trova|bossa nova|samba|\bmpb\b|sertanejo|agronejo|pagode|forr[oó]|"
               r"axé|piseiro|arrocha|seresta|brega|funk carioca|funk de bh|funk consciente|funk melody|"
               r"funk bruxaria|brazilian (funk|pop|bass)|trap funk|dembow|neoperreo|\brkt\b|turreo|urbano|"
               r"música mexicana|flamenco|\bpunta\b|candombe|murga|guajiro|cumbia|techengue|baile funk|regional mexicano|tropical|afro-cuban|kompa|"
               r"cueca|chacarera|marinera|batucada|lambada|maracatu|guaguancó"),
    ("metal", r"metal|deathcore|metalcore|grind|mathcore|djent|rapcore"),
    ("hip hop", r"hip.?hop|\brap\b|\btrap\b|drill|grime|boom bap|phonk|crunk|hyphy|g-funk|horrorcore|"
                r"\bbounce$|memphis|turntablism|nerdcore|chopped and screwed"),
    ("electrónica", r"house|techno|tekno|trance|\bedm|electro\b|electroclash|electronic|dubstep|step\b|"
                    r"drum (and|&|n) bass|jungle|break(beat|core|s\b)|ambient|\bidm\b|uk garage|^garage$|"
                    r"uk funky|\bbass\b|bassline|hardstyle|happy hardcore|gabber|speedcore|frenchcore|"
                    r"hardcore techno|synthwave|vaporwave|chillwave|downtempo|trip hop|lo-fi|glitch|"
                    r"big beat|big room|footwork|\bclub\b|eurodance|italo d|hi-nrg|nu disco|indie dance|"
                    r"alternative dance|^dance$|\bebm\b|darkwave|cold wave|witch house|new age|^drone|"
                    r"nightcore|moombahton|liquid funk|new rave|^industrial$|psytrance|future bass|"
                    r"\bdnb\b|chiptune|electronica|edm|funkot|bacardi|epadunk|j-dance|space music|"
                    r"darksynth|dreamwave|deep tech|hi-tech|frenchtek|minimal wave"),
    ("reggae", r"reggae|dancehall|\bska\b|\bdub\b|rocksteady|ragga|riddim|lovers rock|soca|calypso|shatta"),
    ("africana", r"afro|amapiano|gqom|highlife|hiplife|bongo flava|kuduro|kizomba|zouk|azonto|alté|"
                 r"asakaa|singeli|gengetone|maskandi|fújì|fuji|ndombolo|rumba congolaise|coupé décalé|"
                 r"bikutsi|rap ivoire|gnawa|ethiopian|mbalax|soukous|kwaito|juju"),
    ("jazz", r"jazz|swing music|^swing$|bebop|\bbop\b|big band|ragtime|boogie-woogie|lounge|exotica|easy listening|"
             r"adult standards|dixieland"),
    ("blues", r"blues"),
    ("country", r"country|bluegrass|americana|honky.?tonk|red dirt|newgrass|cajun|zydeco"),
    ("clásica", r"classical|opera|choral|chamber music|orchestra|requiem|minimalism|^medieval|"
                r"renaissance|baroque$|^symphon|romantic era|classical piano|private school piano"),
    ("bandas sonoras", r"soundtrack|\bscore\b|musicals?\b|show tunes|vgm|video game|^anime$|film"),
    ("folk", r"folk|singer-songwriter|celtic|traditional music|shant|acoustic folk"),
    ("pop", r"disco polo"),  # RE2 no tiene lookahead: se saca antes de que "disco" lo mande a soul
    ("r&b/soul", r"r&b|rnb|rhythm and blues|soul|funk|disco|motown|doo-wop|new jack swing|"
                 r"quiet storm|go-go|^boogie$"),
    ("rock", r"rock|punk|hardcore|\bemo|screamo|grunge|shoegaze|new wave|britpop|madchester|psychobilly|"
             r"rockabilly|slowcore|\baor\b|riot grrrl|queercore|visual kei|jam band|surf|power pop|"
             r"jangle pop|neue deutsche welle|^indie$|post-punk|deathrock|noise|"
             r"skiffle|freakbeat|grebo|dark cabaret"),
    ("pop", r"pop|ballad|vocaloid|\bopm\b|variété|shibuya-kei|schlager|chanson|europop|dansband|iskelmä|"
            r"cantopop|mandopop|enka|kayokyoku|dansktop|canzone|neomelodico|laïko|entehno|disco polo|yé-yé"),
    ("world", r"bollywood|tollywood|kollywood|mollywood|sandalwood|bhojiwood|desi|punjabi|bhangra|"
              r"haryanvi|gujarati|bhojpuri|marathi|kannada|telugu|tamil|malayalam|bangla|hindi|ghazal|"
              r"arab|khaleeji|mahraganat|shaabi|chaabi|raï|egyptian|moroccan|turkish|oyun havasi|"
              r"karadeniz|anatolian|mizrahi|fado|polka|manele|dangdut|koplo|funkot|lagu|batak|maluku|"
              r"gufeng|luk thung|phleng|indian|bhojpuri|carnatic|hindustani|klezmer|balkan|gamelan|"
              r"native american|kundiman|harana|budots|indonesian|malay|thai|vietnam|japanese|chinese|"
              r"korean|world"),
    ("rock", r"indie"),  # lo indie que no casó antes (japanese indie va a world, latin indie a latina)
]
