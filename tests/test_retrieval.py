"""
Retrieval ground-truth evaluation — 109 query scenarios across all 30 categories.

Queries are written in realistic form: terse, fragmented, lowercase, the way
students and help-desk agents actually type — NOT polished full sentences.

Each test case:
  query        : realistic user phrasing (short, messy, natural)
  expected_ids : KB article ID(s) that must appear in top-k results
                 Multiple IDs = any one of them is an acceptable answer
  category     : topic area for per-category breakdown
  description  : what is being tested

Metrics:
  Recall@k — at least 1 expected_id in the top-k returned articles

Run:
    python tests/test_retrieval.py            # full evaluation report
    python tests/test_retrieval.py --top-k 3  # stricter — recall@3
    pytest tests/test_retrieval.py -v         # individual pass/fail
    pytest tests/test_retrieval.py -k "VPN"   # filter by category
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Ground-truth test cases  (realistic query phrasing throughout)
TEST_CASES = [
    # ── NetID ────────────────────────────────────────────────────────────────
    {
        "id": "N01", "category": "NetID",
        "query": "netid activation new account",
        "expected_ids": ["1140"],
        "description": "First-time NetID activation",
    },
    {
        "id": "N02", "category": "NetID",
        "query": "forgot netid password reset",
        "expected_ids": ["4610"],
        "description": "NetID password reset / account recovery",
    },
    {
        "id": "N03", "category": "NetID",
        "query": "change netid password",
        "expected_ids": ["20589"],
        "description": "Changing an existing NetID password",
    },
    {
        "id": "N04", "category": "NetID",
        "query": "account locked too many wrong passwords",
        "expected_ids": ["25365"],
        "description": "Too many failed login attempts lockout",
    },
    {
        "id": "N05", "category": "NetID",
        "query": "forgot my netid username",
        "expected_ids": ["51193"],
        "description": "Recovering forgotten NetID username",
    },
    {
        "id": "N06", "category": "NetID",
        "query": "add recovery email to netid account",
        "expected_ids": ["51194"],
        "description": "Adding NetID recovery email",
    },

    # ── Duo MFA ───────────────────────────────────────────────────────────────
    {
        "id": "M01", "category": "Duo_MFA",
        "query": "duo setup new phone first time",
        "expected_ids": ["81448"],
        "description": "First-time Duo smartphone setup",
    },
    {
        "id": "M02", "category": "Duo_MFA",
        "query": "lost phone cant login duo",
        "expected_ids": ["81983"],
        "description": "Lost/broken device — Duo bypass",
    },
    {
        "id": "M03", "category": "Duo_MFA",
        "query": "need temp passcode duo cant use phone",
        "expected_ids": ["87569"],
        "description": "Requesting a temporary Duo passcode",
    },
    {
        "id": "M04", "category": "Duo_MFA",
        "query": "what is duo mfa uw madison",
        "expected_ids": ["81456"],
        "description": "Explaining what MFA/Duo is",
    },
    {
        "id": "M05", "category": "Duo_MFA",
        "query": "got new phone duo not working reactivate",
        "expected_ids": ["88607"],
        "description": "Reactivating Duo on a new device",
    },
    {
        "id": "M06", "category": "Duo_MFA",
        "query": "duo mfa traveling no service abroad",
        "expected_ids": ["85208"],
        "description": "Duo usage while traveling abroad",
    },
    {
        "id": "M07", "category": "Duo_MFA",
        "query": "duo backup passcodes generate",
        "expected_ids": ["80795"],
        "description": "Generating Duo backup passcodes",
    },

    # ── VPN ───────────────────────────────────────────────────────────────────
    {
        "id": "V01", "category": "VPN",
        "query": "wiscvpn install windows",
        "expected_ids": ["148522"],
        "description": "WiscVPN Windows installation",
    },
    {
        "id": "V02", "category": "VPN",
        "query": "globalprotect mac setup",
        "expected_ids": ["148527"],
        "description": "WiscVPN macOS installation",
    },
    {
        "id": "V03", "category": "VPN",
        "query": "vpn not connecting windows fix",
        "expected_ids": ["84893", "1981"],
        # 84893 = Windows-specific troubleshooting; 1981 = general troubleshooting
        # both are valid answers — 1981 consistently surfaces for this query
        "description": "VPN connectivity troubleshooting Windows",
    },
    {
        "id": "V04", "category": "VPN",
        "query": "wiscvpn iphone install ios",
        "expected_ids": ["148595"],
        "description": "WiscVPN iOS setup",
    },
    {
        "id": "V05", "category": "VPN",
        "query": "globalprotect android setup",
        "expected_ids": ["148582"],
        "description": "WiscVPN Android setup",
    },
    {
        "id": "V06", "category": "VPN",
        "query": "uninstall globalprotect mac",
        "expected_ids": ["75258"],
        "description": "Uninstalling WiscVPN macOS",
    },
    {
        "id": "V07", "category": "VPN",
        "query": "engineering vpn paloalto setup",
        "expected_ids": ["84859"],
        "description": "College of Engineering VPN",
    },

    # ── WiFi ──────────────────────────────────────────────────────────────────
    {
        "id": "W01", "category": "WiFi",
        "query": "connect to campus wifi uw",
        "expected_ids": ["25020"],
        "description": "Connecting to campus wireless",
    },
    {
        "id": "W02", "category": "WiFi",
        "query": "eduroam not connecting",
        "expected_ids": ["152453"],
        "description": "eduroam troubleshooting",
    },
    {
        "id": "W03", "category": "WiFi",
        "query": "uwnet internet not working",
        "expected_ids": ["9727"],
        "description": "UWNet general troubleshooting",
    },
    {
        "id": "W04", "category": "WiFi",
        "query": "login to uwnet",
        "expected_ids": ["22915"],
        "description": "UWNet login",
    },
    {
        "id": "W05", "category": "WiFi",
        "query": "guest wifi account visitor campus",
        "expected_ids": ["149853"],
        "description": "Creating UWNet guest account",
    },

    # ── Canvas ────────────────────────────────────────────────────────────────
    {
        "id": "C01", "category": "Canvas",
        "query": "canvas lms getting started uw",
        "expected_ids": ["121975", "62630"],
        "description": "Canvas getting started / overview",
    },
    {
        "id": "C02", "category": "Canvas",
        "query": "canvas storage quota files",
        "expected_ids": ["66526"],
        "description": "Canvas file storage quota",
    },
    {
        "id": "C03", "category": "Canvas",
        "query": "canvas student faq",
        "expected_ids": ["93957", "62630"],
        # 93957 = dedicated student FAQ; 62630 = overview (also answers FAQ queries)
        "description": "Canvas student FAQ",
    },
    {
        "id": "C04", "category": "Canvas",
        "query": "canvas quiz best practices online exam",
        "expected_ids": ["101386"],
        "description": "Canvas quiz best practices",
    },
    {
        "id": "C05", "category": "Canvas",
        "query": "canvas course access reports",
        "expected_ids": ["103854"],
        "description": "Canvas course access reports",
    },

    # ── O365 / Email ──────────────────────────────────────────────────────────
    {
        "id": "O01", "category": "O365",
        "query": "m365 access removed affiliation change",
        "expected_ids": ["62576"],
        "description": "M365 access reduction due to affiliation change",
    },
    {
        "id": "O02", "category": "O365",
        "query": "setup outlook windows uw email",
        "expected_ids": ["52197"],
        "description": "Outlook Windows configuration",
    },
    {
        "id": "O03", "category": "O365",
        "query": "shared mailbox setup office365",
        "expected_ids": ["157020"],
        "description": "Shared mailbox setup",
    },
    {
        "id": "O04", "category": "O365",
        "query": "leaving uw what happens to o365 account",
        "expected_ids": ["79454", "80255"],
        "description": "Departing student/staff M365 deactivation",
    },
    {
        "id": "O05", "category": "O365",
        "query": "alumni email access after graduation",
        "expected_ids": ["6023"],
        "description": "Alumni email access",
    },

    # ── Zoom ──────────────────────────────────────────────────────────────────
    {
        "id": "Z01", "category": "Zoom",
        "query": "zoom getting started uw madison",
        "expected_ids": ["105271"],
        "description": "Zoom getting started",
    },
    {
        "id": "Z02", "category": "Zoom",
        "query": "zoom instructor faq teaching online",
        "expected_ids": ["110758"],
        "description": "Zoom instructor FAQ",
    },
    {
        "id": "Z03", "category": "Zoom",
        "query": "download zoom recording cloud",
        "expected_ids": ["109403"],
        "description": "Managing Zoom recordings",
    },
    {
        "id": "Z04", "category": "Zoom",
        "query": "zoom meeting security settings prevent bombing",
        "expected_ids": ["106947"],
        "description": "Zoom meeting security",
    },
    {
        "id": "Z05", "category": "Zoom",
        "query": "zoom ai companion how to use",
        "expected_ids": ["139097"],
        "description": "Zoom AI Companion feature",
    },

    # ── Adobe CC ──────────────────────────────────────────────────────────────
    {
        "id": "A01", "category": "Adobe_CC",
        "query": "adobe creative cloud login uw account",
        "expected_ids": ["69772"],
        "description": "Adobe CC UW login",
    },
    {
        "id": "A02", "category": "Adobe_CC",
        "query": "adobe creative cloud student subscription",
        "expected_ids": ["105550", "120207"],
        "description": "Student Adobe CC access",
    },
    {
        "id": "A03", "category": "Adobe_CC",
        "query": "all adobe apps say trial not activated",
        "expected_ids": ["99745"],
        "description": "Adobe apps showing as trial",
    },
    {
        "id": "A04", "category": "Adobe_CC",
        "query": "adobe acrobat uw getting started",
        "expected_ids": ["62176"],
        "description": "Adobe Acrobat getting started",
    },

    # ── Google Workspace ──────────────────────────────────────────────────────
    {
        "id": "G01", "category": "Google_Workspace",
        "query": "google workspace login uw",
        "expected_ids": ["17276"],
        "description": "Google Workspace login",
    },
    {
        "id": "G02", "category": "Google_Workspace",
        "query": "share file google drive uw",
        "expected_ids": ["14067"],
        "description": "Sharing Google Drive files",
    },
    {
        "id": "G03", "category": "Google_Workspace",
        "query": "upload file google drive",
        "expected_ids": ["13683"],
        "description": "Uploading to Google Drive",
    },
    {
        "id": "G04", "category": "Google_Workspace",
        "query": "am i eligible google workspace uw",
        "expected_ids": ["47616"],
        "description": "Google Workspace eligibility",
    },

    # ── Teams ─────────────────────────────────────────────────────────────────
    {
        "id": "T01", "category": "Teams",
        "query": "teams cache clear slow crashing",
        "expected_ids": ["110603"],
        "description": "Clearing Teams client cache",
    },
    {
        "id": "T02", "category": "Teams",
        "query": "join teams meeting as guest",
        "expected_ids": ["104833"],
        "description": "Joining Teams as guest",
    },
    {
        "id": "T03", "category": "Teams",
        "query": "teams broken in chrome browser",
        "expected_ids": ["114600"],
        "description": "Teams Chrome sandboxing issue",
    },

    # ── Cloud Storage ─────────────────────────────────────────────────────────
    {
        "id": "CS01", "category": "Cloud_Storage",
        "query": "onedrive storage quota how much space",
        "expected_ids": ["159933"],
        "description": "OneDrive storage quota",
    },
    {
        "id": "CS02", "category": "Cloud_Storage",
        "query": "access onedrive for business",
        "expected_ids": ["46186", "46143"],
        "description": "Accessing OneDrive for Business",
    },
    {
        "id": "CS03", "category": "Cloud_Storage",
        "query": "send large files not email attachment",
        "expected_ids": ["42214"],
        "description": "Large file sharing alternatives",
    },
    {
        "id": "CS04", "category": "Cloud_Storage",
        "query": "move data from box to researchdrive",
        "expected_ids": ["102788"],
        "description": "Box to ResearchDrive data transfer",
    },

    # ── Office Install ────────────────────────────────────────────────────────
    {
        "id": "OI01", "category": "Office_Install",
        "query": "download microsoft office 365 personal computer",
        "expected_ids": ["43841"],
        "description": "Microsoft Office 365 download and install",
    },
    {
        "id": "OI02", "category": "Office_Install",
        "query": "install office from 365 portal",
        "expected_ids": ["72149"],
        "description": "Installing Office from O365 portal",
    },
    {
        "id": "OI03", "category": "Office_Install",
        "query": "campus software library how to download",
        "expected_ids": ["36603"],
        "description": "Campus Software Library download",
    },
    {
        "id": "OI04", "category": "Office_Install",
        "query": "office 365 apps in browser",
        "expected_ids": ["132544"],
        "description": "Microsoft 365 web browser access",
    },

    # ── Printing ──────────────────────────────────────────────────────────────
    {
        "id": "PR01", "category": "Printing",
        "query": "pay to print mobile campus",
        "expected_ids": ["131627"],
        "description": "Pay-for-print and mobile printing",
    },
    {
        "id": "PR02", "category": "Printing",
        "query": "printing science hall",
        "expected_ids": ["121493"],
        "description": "Printing in Science Hall",
    },
    {
        "id": "PR03", "category": "Printing",
        "query": "print at cae engineering",
        "expected_ids": ["27894"],
        "description": "CAE printing",
    },

    # ── Remote Desktop ────────────────────────────────────────────────────────
    {
        "id": "RD01", "category": "Remote_Desktop",
        "query": "rdp into windows computer remote",
        "expected_ids": ["125472"],
        "description": "RDP into Windows computer",
    },
    {
        "id": "RD02", "category": "Remote_Desktop",
        "query": "what is rds remote desktop service uw",
        "expected_ids": ["70798"],
        "description": "RDS overview",
    },
    {
        "id": "RD03", "category": "Remote_Desktop",
        "query": "remote work tools access campus resources",
        "expected_ids": ["10038"],
        "description": "Remote working tools overview",
    },
    {
        "id": "RD04", "category": "Remote_Desktop",
        "query": "azure remote pc access",
        "expected_ids": ["134285"],
        "description": "Azure Remote PC access",
    },

    # ── Phishing / Email Security ─────────────────────────────────────────────
    {
        "id": "PH01", "category": "Phishing",
        "query": "got phishing email what to do",
        "expected_ids": ["52781"],
        "description": "Phishing detection and remediation",
    },
    {
        "id": "PH02", "category": "Phishing",
        "query": "url defense links rewritten in email",
        "expected_ids": ["132650"],
        "description": "URL Defense email security",
    },
    {
        "id": "PH03", "category": "Phishing",
        "query": "junk mail phishing office 365",
        "expected_ids": ["31866"],
        "description": "O365 junk/phishing email",
    },
    {
        "id": "PH04", "category": "Phishing",
        "query": "account hacked via phishing help",
        "expected_ids": ["107115"],
        "description": "Compromised credentials via phishing",
    },

    # ── Antivirus / Security ──────────────────────────────────────────────────
    {
        "id": "AV01", "category": "Antivirus",
        "query": "run virus scan windows cisco",
        "expected_ids": ["123986"],
        "description": "Cisco Secure Endpoint antivirus scan Windows",
    },
    {
        "id": "AV02", "category": "Antivirus",
        "query": "computer security checklist uw",
        "expected_ids": ["15096"],
        "description": "Computer security checklist",
    },
    {
        "id": "AV03", "category": "Antivirus",
        "query": "virus malware on my laptop",
        "expected_ids": ["9974"],
        "description": "Malware/virus infection on machine",
    },

    # ── Kaltura ───────────────────────────────────────────────────────────────
    {
        "id": "K01", "category": "Kaltura",
        "query": "embed kaltura video in canvas page",
        "expected_ids": ["107750"],
        "description": "Embedding Kaltura video in Canvas",
    },
    {
        "id": "K02", "category": "Kaltura",
        "query": "upload share video canvas kaltura",
        "expected_ids": ["63003"],
        "description": "Upload video to Canvas via Kaltura",
    },
    {
        "id": "K03", "category": "Kaltura",
        "query": "add quiz to kaltura video",
        "expected_ids": ["60958"],
        "description": "Kaltura video quiz feature",
    },

    # ── VoIP / Phones ─────────────────────────────────────────────────────────
    {
        "id": "VP01", "category": "VoIP",
        "query": "change voicemail pin campus phone",
        "expected_ids": ["72402"],
        "description": "Changing VoIP voicemail PIN",
    },
    {
        "id": "VP02", "category": "VoIP",
        "query": "access voicemail cisco campus phone",
        "expected_ids": ["72655"],
        "description": "Accessing campus phone voicemail",
    },
    {
        "id": "VP03", "category": "VoIP",
        "query": "cisco voip phone faq campus",
        "expected_ids": ["149678"],
        "description": "Cisco VoIP FAQ",
    },

    # ── ResearchDrive ─────────────────────────────────────────────────────────
    {
        "id": "RR01", "category": "ResearchDrive",
        "query": "researchdrive faq",
        "expected_ids": ["95074"],
        "description": "ResearchDrive FAQ",
    },
    {
        "id": "RR02", "category": "ResearchDrive",
        "query": "researchdrive who is eligible free",
        "expected_ids": ["129905"],
        "description": "ResearchDrive eligibility",
    },
    {
        "id": "RR03", "category": "ResearchDrive",
        "query": "transfer google drive to researchdrive",
        "expected_ids": ["127009"],
        "description": "Google to ResearchDrive transfer",
    },

    # ── Endpoint Management ───────────────────────────────────────────────────
    {
        "id": "EM01", "category": "Endpoint_Mgmt",
        "query": "workspace one macos enrollment",
        "expected_ids": ["132963"],
        "description": "Workspace ONE macOS enrollment",
    },
    {
        "id": "EM02", "category": "Endpoint_Mgmt",
        "query": "workspace one windows student enroll",
        "expected_ids": ["132962"],
        "description": "Workspace ONE Windows student enrollment",
    },

    # ── Active Directory ──────────────────────────────────────────────────────
    {
        "id": "AD01", "category": "Active_Directory",
        "query": "join windows to campus active directory",
        "expected_ids": ["34872"],
        "description": "Joining Windows to Campus AD",
    },
    {
        "id": "AD02", "category": "Active_Directory",
        "query": "connect mac to campus ad",
        "expected_ids": ["28637"],
        "description": "Joining Mac to Campus AD",
    },
    {
        "id": "AD03", "category": "Active_Directory",
        "query": "campus active directory overview",
        "expected_ids": ["12331"],
        "description": "Active Directory overview",
    },

    # ── Data Policy ───────────────────────────────────────────────────────────
    {
        "id": "DP01", "category": "Data_Policy",
        "query": "sensitive data allowed in cloud uw",
        "expected_ids": ["110947"],
        "description": "Cloud eligibility for sensitive/restricted data",
    },
    {
        "id": "DP02", "category": "Data_Policy",
        "query": "what data can go in public cloud",
        "expected_ids": ["100124"],
        "description": "Data elements allowed in public cloud",
    },
    {
        "id": "DP03", "category": "Data_Policy",
        "query": "aws restricted data uw research",
        "expected_ids": ["115304"],
        "description": "AWS for sensitive/restricted data",
    },

    # ── Stats Software ────────────────────────────────────────────────────────
    {
        "id": "SS01", "category": "Stats_Software",
        "query": "install sas windows uw",
        "expected_ids": ["114986", "60445"],
        "description": "SAS Windows installation",
    },
    {
        "id": "SS02", "category": "Stats_Software",
        "query": "sas installation guide uw",
        "expected_ids": ["60445"],
        "description": "SAS installation guide",
    },

    # ── Software Library ──────────────────────────────────────────────────────
    {
        "id": "SL01", "category": "Software_Library",
        "query": "download install spss campus library",
        "expected_ids": ["37652", "73473"],
        "description": "SPSS download and install",
    },
    {
        "id": "SL02", "category": "Software_Library",
        "query": "update spss license code",
        "expected_ids": ["48505"],
        "description": "Updating SPSS license code",
    },
    {
        "id": "SL03", "category": "Software_Library",
        "query": "spss error codes fix",
        "expected_ids": ["62753"],
        "description": "SPSS error codes",
    },

    # ── Computer Repair ───────────────────────────────────────────────────────
    {
        "id": "CR01", "category": "Computer_Repair",
        "query": "borrow loaner laptop doit",
        "expected_ids": ["110576"],
        "description": "DoIT laptop loaner program",
    },
    {
        "id": "CR02", "category": "Computer_Repair",
        "query": "computer repair doit service",
        "expected_ids": ["101647"],
        "description": "DoIT computer repair services",
    },

    # ── Student Center / SIS ──────────────────────────────────────────────────
    {
        "id": "SC01", "category": "Student_Center",
        "query": "sis access request getting started",
        "expected_ids": ["121046"],
        "description": "SIS getting started and access request",
    },
    {
        "id": "SC02", "category": "Student_Center",
        "query": "sis student records lookup",
        "expected_ids": ["1794"],
        "description": "SIS student records inquiry",
    },

    # ── Web Hosting ───────────────────────────────────────────────────────────
    {
        "id": "WH01", "category": "Web_Hosting",
        "query": "wordpress web hosting uw doit",
        "expected_ids": ["33416"],
        "description": "DoIT WordPress web hosting",
    },
    {
        "id": "WH02", "category": "Web_Hosting",
        "query": "web hosting maintenance window",
        "expected_ids": ["29535"],
        "description": "Web hosting maintenance windows",
    },

    # ── OS Support ───────────────────────────────────────────────────────────
    {
        "id": "OS01", "category": "OS_Support",
        "query": "upgrade to macos sequoia",
        "expected_ids": ["147320"],
        "description": "Upgrading to macOS Sequoia",
    },
    {
        "id": "OS02", "category": "OS_Support",
        "query": "browser not loading troubleshoot",
        "expected_ids": ["109511"],
        "description": "Browser troubleshooting",
    },
    {
        "id": "OS03", "category": "OS_Support",
        "query": "install windows on mac boot camp",
        "expected_ids": ["20569"],
        "description": "Boot Camp Windows on Mac",
    },
    {
        "id": "OS04", "category": "OS_Support",
        "query": "macos update faculty staff computers",
        "expected_ids": ["135360"],
        "description": "Faculty/staff macOS OS updates",
    },

    # ── Classroom AV ──────────────────────────────────────────────────────────
    {
        "id": "CA01", "category": "Classroom_AV",
        "query": "classroom av support help",
        "expected_ids": ["45982"],
        "description": "Classroom media support",
    },
    {
        "id": "CA02", "category": "Classroom_AV",
        "query": "classroom recording equipment lecture setup",
        "expected_ids": ["98806"],
        "description": "Classroom recording equipment setup",
    },

    # ── Webex ─────────────────────────────────────────────────────────────────
    {
        "id": "WX01", "category": "Webex",
        "query": "cisco webex conference room getting started",
        "expected_ids": ["94356"],
        "description": "Webex/Cisco video conferencing getting started",
    },
    {
        "id": "WX02", "category": "Webex",
        "query": "hybrid conference room doit cisco",
        "expected_ids": ["114941"],
        "description": "DoIT hybrid conference rooms",
    },
]


# Evaluation runner
def run_evaluation(top_k: int = 5, verbose: bool = True) -> dict:
    try:
        from retriever import retrieve
    except ImportError as e:
        raise SystemExit(f"Cannot import retriever: {e}")

    results = []
    cat_stats = defaultdict(lambda: {"pass": 0, "fail": 0})

    for tc in TEST_CASES:
        try:
            hits = retrieve(tc["query"], top_k=top_k)
            returned_ids = {h["id"] for h in hits}
        except Exception as e:
            returned_ids = set()
            if verbose:
                print(f"  ⚠  {tc['id']} retrieve() raised: {e}")

        expected = set(tc["expected_ids"])
        passed = bool(expected & returned_ids)

        results.append({
            "id":           tc["id"],
            "category":     tc["category"],
            "query":        tc["query"],
            "expected_ids": tc["expected_ids"],
            "returned_ids": list(returned_ids),
            "passed":       passed,
            "description":  tc["description"],
        })
        cat_stats[tc["category"]]["pass" if passed else "fail"] += 1

    n_total  = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    recall   = n_passed / n_total * 100

    if verbose:
        _print_report(results, cat_stats, n_total, n_passed, recall, top_k)

    return {
        "results":     results,
        "cat_stats":   dict(cat_stats),
        "recall_at_k": recall,
        "n_total":     n_total,
        "n_passed":    n_passed,
        "top_k":       top_k,
    }


def _print_report(results, cat_stats, n_total, n_passed, recall, top_k):
    SEP = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  DoIT KB Retrieval Evaluation — Recall@{top_k}")
    print(f"{'═'*72}")
    print(f"  Total queries : {n_total}")
    print(f"  Passed        : {n_passed}  ({recall:.1f}%)")
    print(f"  Failed        : {n_total - n_passed}  ({100-recall:.1f}%)")

    print(f"\n{SEP}")
    print(f"  {'Category':<22}  {'Pass':>4}  {'Fail':>4}  {'Rate':>6}")
    print(SEP)
    for cat in sorted(cat_stats):
        p = cat_stats[cat]["pass"]
        f = cat_stats[cat]["fail"]
        rate = p / (p + f) * 100
        bar = "✓" if f == 0 else ("△" if rate >= 50 else "✗")
        print(f"  {bar} {cat:<20}  {p:>4}  {f:>4}  {rate:>5.0f}%")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{SEP}")
        print(f"  FAILURES ({len(failures)} cases):")
        print(SEP)
        for r in failures:
            print(f"\n  [{r['id']}] {r['category']} — {r['description']}")
            print(f"  Query   : {r['query']}")
            print(f"  Expected: {r['expected_ids']}")
            print(f"  Got     : {sorted(r['returned_ids'])}")
    else:
        print(f"\n  ✓  All tests passed — perfect recall@{top_k}!")

    print(f"\n{'═'*72}\n")


# pytest interface
def _make_pytest_test(tc):
    import pytest

    @pytest.mark.parametrize("top_k", [5])
    def test_fn(top_k):
        from retriever import retrieve
        hits = retrieve(tc["query"], top_k=top_k)
        returned = {h["id"] for h in hits}
        expected = set(tc["expected_ids"])
        assert expected & returned, (
            f"\n[{tc['id']}] {tc['description']}\n"
            f"Query   : {tc['query']}\n"
            f"Expected: {tc['expected_ids']}\n"
            f"Got     : {sorted(returned)}"
        )

    test_fn.__name__ = f"test_{tc['id'].lower()}_{tc['category'].lower()}"
    test_fn.__doc__  = f"[{tc['category']}] {tc['description']}"
    return test_fn


for _tc in TEST_CASES:
    _fn_name = f"test_{_tc['id'].lower()}_{_tc['category'].lower()}"
    globals()[_fn_name] = _make_pytest_test(_tc)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DoIT KB retrieval evaluation")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to retrieve per query (default: 5)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a specific category (e.g. VPN)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print summary, suppress failure details")
    args = parser.parse_args()

    if args.category:
        original = TEST_CASES[:]
        filtered = [tc for tc in TEST_CASES
                    if tc["category"].lower() == args.category.lower()]
        if not filtered:
            print(f"No test cases for category '{args.category}'.")
            sys.exit(1)
        TEST_CASES[:] = filtered
        run_evaluation(top_k=args.top_k, verbose=not args.quiet)
        TEST_CASES[:] = original
    else:
        run_evaluation(top_k=args.top_k, verbose=not args.quiet)
