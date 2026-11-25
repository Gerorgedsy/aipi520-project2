"""Step 2a - rule-based labeling from overall_status + why_stopped.

    label = 1  the trial ran to completion (or stopped early for success)
    label = 0  the trial was withdrawn/terminated for a reason that reflects on
               the trial itself (safety, lack of efficacy, could not enroll,
               or no usable reason given)
    label = NA stopped for a reason unrelated to the trial's merit (business
               decision, funding, logistics, COVID) -> excluded from modeling

This is a proxy for "success": a completed trial may still have missed its
endpoint. The rules are deliberately transparent so they can be audited:
`python -m src.audit_labels` prints random examples per category.
"""
import re

# Phrases that mention a topic only to rule it out ("not due to safety concerns",
# "no safety issues", "isn't related to any safety/efficacy concern"). They are
# removed before matching so they cannot trigger the category they deny.
_TOPIC = r"(?:safety|efficacy|adverse|toxicit\w*|tolerab\w*|futility|side effects?|benefit[\s/-]*risk|risk[\s/-]*benefit)"
_TAIL = (r"(?:[\s/,]*(?:or|and|nor)?[\s/]*(?:" + _TOPIC + r"|quality|clinical activity|concerns?|issues?|"
         r"reasons?|findings?|signals?|problems?|events?|related|decision|data))*")
_NEGATED = re.compile(
    # "not due to / not related to / not based on ... safety or efficacy concerns"
    r"(?:\bnot|n't|\bnor|\bneither)\s+(?:been\s+)?(?:due|related|based|linked|because|driven|a consequence|"
    r"for reasons?|as a result|a result|associated|caused|prompted|motivated|stopped (?:due|for))\b"
    r"[^.;]{0,70}?\b" + _TOPIC + r"\b" + _TAIL +
    # "not a safety decision", "not safety related"
    r"|\bnot\s+(?:an?\s+)?" + _TOPIC + r"\b" + _TAIL +
    # "no (new) safety concerns", "without safety issues" - safety-type topics only,
    # because "no efficacy" is a real efficacy failure, not a denial.
    r"|\b(?:no|without|absence of|unrelated to|rather than)\s+(?:any\s+|new\s+|known\s+|identified\s+|significant\s+)*"
    r"(?:safety|adverse|toxicit\w*|tolerab\w*|side effects?)\b" + _TAIL,
    re.IGNORECASE,
)

# Words that turn a "success" phrase into its opposite ("did NOT meet", "LACK of efficacy").
_NEGATIVE_CUE = re.compile(
    r"\b(?:not|no|never|lack\w*|fail\w*|insufficient|unable|futil\w*|low|poor|without|"
    r"unlikely|inadequate|limited|modest)\b|n't",
    re.IGNORECASE,
)

# "safety" used neutrally ("safety study", "safety data", "safety endpoint") is not a safety stop.
_SAFETY = (r"safety(?!\s+(?:stud|endpoint|data|review|follow|monitor|analys|assess|committee|"
           r"run|lead|cohort|population|and efficacy|results|profile|evaluation|objective))")

# First match wins, so the order encodes priority: evidence about the drug itself
# (success, efficacy, safety) outranks circumstances (COVID, enrollment, money, logistics).
CATEGORY_PATTERNS = [
    # Ended early because it worked, or "terminated" although it actually finished.
    # Only applied when the text has no negative cue (see stop_category).
    ("positive_early",
     r"met (?:its |the |all )?(?:primary|pre-?specified|efficacy|accrual)|"
     r"efficacy (?:was |has been )?(?:demonstrated|established|shown|achieved)|"
     r"overwhelming(?:ly)? (?:efficacy|benefit|positive)|superiority|"
     r"early (?:success|efficacy)|positive (?:interim|results|outcome)|"
     r"(?:fda|ema|regulatory|marketing) approval (?:was |has been )?(?:obtained|granted|received)|"
     r"approved by|(?:drug|product) (?:was |is |got |received )?(?:approved|licensed)|"
     r"became (?:commercially )?available|"
     r"(?:objectives?|goals?|endpoints?|target|sample size)\)? (?:were |was |had been |has been )?(?:met|achieved|fulfilled|reached)|"
     r"reach(?:ed|ing) (?:the |its )?(?:required|planned|target|pre-?determined|necessary)|"
     r"achievement of|concluded as planned|completed (?:per|as|according)|"
     r"^(?:the )?(?:study |trial )?(?:was |is |has been )?(?:administratively )?complete[d]?\.?$|"
     r"sufficient data|enough data|acceptable to meet"),
    ("efficacy_futility",
     r"futil|lack(?:s|ed|ing)? (?:of )?(?:efficacy|effect|benefit|response|activity)|"
     r"(?:no|absence of)\b[^.;]{0,30}(?:efficacy|benefit|effect|activity|difference|improve|response|advantage)|"
     r"not (?:effective|efficacious|be adequate|adequate)|ineffective|inefficac|effectiveness|"
     r"insufficient (?:efficacy|activity|response|benefit|effect)|"
     r"(?:modest|limited|low|poor|marginal|minimal|weak)\w* (?:clinical |anti-?tumou?r |therapeutic )?(?:benefit|efficacy|activity|response)|"
     r"(?:efficacy|activity|response|benefit)[^.;]{0,50}(?:not sufficient|insufficient|low|poor|limited|disappointing|not (?:be )?(?:achieved|met|adequate))|"
     r"(?:lower|less|worse) than (?:estimated|expected|anticipated)[^.;]{0,25}(?:efficacy|response|benefit|activity)|"
     r"(?:did|does|would|could|will) not (?:meet|reach|achieve|show|demonstrate|offer)|"
     r"fail(?:ed|ure)? to (?:meet|reach|achieve|show|demonstrate)|unlikely to (?:meet|reach|achieve|show|demonstrate)|"
     r"(?:criteria|endpoints?)[^.;]{0,20}not (?:met|reached|achieved)|"
     r"negative (?:results|outcome|interim|findings)|stopping (?:rule|boundar)|"
     r"interim analys[ie]s|intermediary analysis|unfavou?rable|not (?:warrant|support)|primary endpoint"),
    ("safety",
     _SAFETY + r"|unsafe|adverse|toxic|side effects?|risk[\s/-]*benefit|benefit[\s/-]*risk|risks? (?:were|was) too|"
     r"\bdeaths?\b|mortality|harm|tolerab|\bsaes?\b|susar|hepatotox|hepatic|liver (?:enzymes|toxicity|injury|safety)|transaminase|cardiotox|"
     r"clinical hold|black box|warning|high incidence|"
     r"\bdsm[bc]\b|\bdmc\b|data (?:and )?safety monitoring|(?:data|safety) monitoring (?:board|committee)"),
    ("covid", r"covid|corona|pandemic|sars[- ]?cov"),
    ("enrollment",
     r"enrol|recruit|accru|inclusion|slow|"
     r"(?:no|few|lack of|insufficient|not enough|too few|small number of|number of)"
     r"[^.;]{0,30}(?:patients?|subjects?|participants?|volunteers?|cases|candidates)|"
     r"(?:patients?|subjects?|participants?)[^.;]{0,30}(?:not want|refus|declin|excluded|not achieved)|"
     r"participation|drop[\s-]?out|feasib|eligible|sample size|screen(?:ing)? fail"),
    ("business_funding",
     r"fund|financ|budget|grant|money|economic|\bcosts?\b|resource|subsid|"
     r"sponsor|business|strateg|commercial|company|corporate|portfolio|"
     r"priorit|acqui|merger|bankrupt|insolven|reorgani[sz]|licens|support|"
     r"(?:development|program(?:me)?s?|project|compound|indication)[^.;]{0,40}"
     r"(?:discontinu|terminat|stopped|halted|ceased|ended|cancel|suspend|on hold|closed)|"
     r"discontinu\w*[^.;]{0,60}(?:development|program|project)|"
     r"focus on|no longer pursu|not (?:to )?(?:proceed|pursue|continue)|decision|decided"),
    ("logistics",
     r"(?:investigator|\bpi\b|physician|coordinator|staff|personnel|provider|team)|"
     r"suppl(?:y|ies)|shortage|unavailab|not available|no longer available|unable to obtain|"
     r"cannot be provided|expir|manufactur|production|stability|investigational product|"
     r"withdrawn from (?:the )?market|recall|logistic|operational|"
     r"clos(?:ure|ed|ing) of (?:the )?(?:site|center|centre|clinic|unit)|site clos|"
     r"\birb\b|ethics|committee|regulat|authorit|approval|health canada|recomm?andation|"
     r"administrativ|technical|contract|agreement|renewal|"
     r"never (?:started|initiated|opened)|not (?:be )?(?:started|initiated|opened)|cancel|"
     r"protocol|design|replaced|superseded|new (?:study|trial)|another (?:study|trial|clinical)|"
     r"(?:other|another|similar|separate) stud|will open as|"
     r"standard of care|guideline|landscape|chang\w* in|\bwar\b|hurricane|earthquake"),
]
_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in CATEGORY_PATTERNS]

# Category -> label. None means "exclude the trial from the modeling set".
LABEL_POLICY = {
    "completed": 1,
    "positive_early": 1,
    "efficacy_futility": 0,
    "safety": 0,
    "enrollment": 0,
    "unknown": 0,           # stopped, but no reason given or none recognized
    "covid": None,
    "business_funding": None,
    "logistics": None,
}


def stop_category(overall_status, why_stopped) -> str:
    """Classify how a trial ended into one of the LABEL_POLICY keys."""
    if str(overall_status or "").upper() == "COMPLETED":
        return "completed"
    text = why_stopped if isinstance(why_stopped, str) else ""
    text = _NEGATED.sub(" ", text).strip()
    if not text:
        return "unknown"
    has_negative_cue = bool(_NEGATIVE_CUE.search(text))
    for name, pattern in _COMPILED:
        if name == "positive_early" and has_negative_cue:
            continue
        if pattern.search(text):
            return name
    return "unknown"


def infer_label(overall_status, why_stopped):
    category = stop_category(overall_status, why_stopped)
    return category, LABEL_POLICY[category]
