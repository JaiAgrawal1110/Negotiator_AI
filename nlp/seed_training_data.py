"""
Seed training data for the negotiation tone classifier.

8 signal classes:
  hesitant, urgent, aggressive, bluffing, collaborative,
  dismissive, enthusiastic, noncommittal

DATA SOURCES:
  [REAL] = extracted directly from real client conversations
           (Girish/HouseOfBeau negotiation, aggressive client screenshots,
           art commission screenshot, Facebook Marketplace screenshot)
  [SYNTH] = synthetic examples written to match the style/register of
            the real ones — same tone, same kind of phrasing patterns

Real examples are marked so you know which ones to weight more heavily
if you move to a weighted loss function later.
"""

LABELS = [
    "hesitant",
    "urgent",
    "aggressive",
    "bluffing",
    "collaborative",
    "dismissive",
    "enthusiastic",
    "noncommittal",
]

SEED_DATA = [

    # ------------------------------------------------------------------ #
    # HESITANT
    # ------------------------------------------------------------------ #
    # [REAL] Girish, after Jai quoted $210/week
    ("This seems extremely high. Let me get back to you on Monday.", "hesitant"),
    # [SYNTH] variations on the same register
    ("I'm not totally sure this fits our budget right now, let me think on it.", "hesitant"),
    ("Hmm, I guess that could work, I just need to check with the team first.", "hesitant"),
    ("I'm a little unsure about committing to that number right now.", "hesitant"),
    ("Maybe we could do that... I'd need to sit with it a bit before deciding.", "hesitant"),
    ("That's a bit more than I was expecting, I don't know.", "hesitant"),
    ("Possibly, but I'm hesitant to lock in without seeing more of your work.", "hesitant"),
    ("I want to say yes but something's holding me back, let me reconsider.", "hesitant"),
    ("I'm on the fence about the timeline and the price together.", "hesitant"),
    ("That number makes me a little nervous if I'm honest.", "hesitant"),
    ("Let me circle back on this once I've spoken to my partner.", "hesitant"),
    ("I'm interested but I'm not 100% certain this is the right fit yet.", "hesitant"),
    ("I need a bit more time before I can commit to something like this.", "hesitant"),

    # ------------------------------------------------------------------ #
    # URGENT
    # ------------------------------------------------------------------ #
    # [REAL] Aggressive client, image 2
    ("I need you to respond to my questions. Asap. And be communicating.", "urgent"),
    # [SYNTH]
    ("I need this done by tomorrow morning, no exceptions.", "urgent"),
    ("We're launching Monday, can you confirm today if you can take this on?", "urgent"),
    ("This is time-sensitive, I need an answer ASAP.", "urgent"),
    ("The team is waiting on this right now, how quickly can you start?", "urgent"),
    ("Can we wrap this up today? We're under serious time pressure.", "urgent"),
    ("I need a decision within the hour if possible.", "urgent"),
    ("We're already behind schedule, every single day counts here.", "urgent"),
    ("This needs to ship by Friday no matter what happens.", "urgent"),
    ("Our investors are expecting an update tomorrow, I need this now.", "urgent"),
    ("Can you get this to me end of day? It's blocking everything else.", "urgent"),
    ("I wouldn't normally rush you but we genuinely have no time left.", "urgent"),
    ("Please respond ASAP, I have a client waiting on my end too.", "urgent"),

    # ------------------------------------------------------------------ #
    # AGGRESSIVE
    # ------------------------------------------------------------------ #
    # [REAL] Aggressive client, image 1
    ("I need productivity throughout your whole shift. I'll be cutting pay rate if standards are not met.", "aggressive"),
    ("I need leads flowing in and constant appointments being booked and deposits made every day.", "aggressive"),
    # [REAL] Aggressive client, image 2
    ("From now on three strike rule going into effect.", "aggressive"),
    ("I see you online.", "aggressive"),
    ("Starting today we are implementing a strict 3-Strike Policy. Strike 1 verbal warning. Strike 2 written warning. Strike 3 termination.", "aggressive"),
    # [REAL] Art commission, image 3
    ("why is that so high? it's just a headshot? and it's not even real art because it's digital lol no offense", "aggressive"),
    ("What? shocked because I told you the cold truth?", "aggressive"),
    # [SYNTH]
    ("That price is honestly ridiculous, you're overcharging me.", "aggressive"),
    ("I'm not paying that. Take it or leave it.", "aggressive"),
    ("You need to drop the price right now or we're done here.", "aggressive"),
    ("Stop wasting my time and just give me a real number.", "aggressive"),
    ("Cut the price in half or I'm walking, simple as that.", "aggressive"),
    ("I expected better from you, this quote is honestly insulting.", "aggressive"),
    ("Final offer. Take it or I find someone else immediately.", "aggressive"),
    ("I don't have patience for this back and forth, just lower it.", "aggressive"),

    # ------------------------------------------------------------------ #
    # BLUFFING
    # ------------------------------------------------------------------ #
    # [REAL] Facebook Marketplace, image 4
    ("How about 80 that's as high as I can go.", "bluffing"),
    # [REAL] Girish using salary comparison as leverage
    ("Our most senior Data Analyst in Delhi earns this amount working full time with 20 years of experience.", "bluffing"),
    ("A fresh graduate working full time will earn around INR 25k a month.", "bluffing"),
    # [SYNTH]
    ("I've got three other freelancers ready to do this for half the price.", "bluffing"),
    ("Honestly we might just build this in-house instead.", "bluffing"),
    ("I could probably find someone cheaper in five minutes on Fiverr.", "bluffing"),
    ("We don't actually need this project that badly, just so you know.", "bluffing"),
    ("There's a guy who quoted me a third of this for the same work.", "bluffing"),
    ("We're talking to a couple of other agencies as backup options.", "bluffing"),
    ("Plenty of other people would jump at this rate, you know.", "bluffing"),
    ("Worst case we just delay the whole launch, it's not critical.", "bluffing"),
    ("My cousin actually does this kind of work too, for less.", "bluffing"),
    ("I've done negotiations like this a hundred times, I know the market.", "bluffing"),

    # ------------------------------------------------------------------ #
    # COLLABORATIVE
    # ------------------------------------------------------------------ #
    # [REAL] Girish moving toward agreement
    ("We shall do 3 months at INR 25k and then switch to INR 30k after 3 months. Please let me know if this works for you.", "collaborative"),
    ("Ok so you have received INR 23800 so INR 1200 short. We shall add this to next month's payment. We didn't know what PayPal takes, now we know.", "collaborative"),
    ("We want you to feel comfortable with the cost. Come to us with a proposal.", "collaborative"),
    ("We can commit to giving you a monthly salary for 6 months.", "collaborative"),
    # [SYNTH]
    ("Let's figure out something that works well for both of us.", "collaborative"),
    ("I really value your expertise, let's find the right middle ground here.", "collaborative"),
    ("Happy to adjust the scope if it helps us land on a fair number.", "collaborative"),
    ("What would make this work well on your end? I'm open to ideas.", "collaborative"),
    ("Let's talk through the budget together and see what's realistic for both sides.", "collaborative"),
    ("I'm open to suggestions, let's problem-solve this together.", "collaborative"),
    ("Tell me what flexibility you have and we'll work around it.", "collaborative"),
    ("I appreciate your time and want to make sure this is fair for everyone.", "collaborative"),
    ("We can definitely find somewhere reasonable, I just need to understand your constraints.", "collaborative"),

    # ------------------------------------------------------------------ #
    # DISMISSIVE
    # ------------------------------------------------------------------ #
    # [REAL] Girish/wife on payment dispute
    ("I don't understand. You want us to pay your taxes too?", "dismissive"),
    ("Normally when we agree on an amount the taxes is on the person receiving it. This is how payments work.", "dismissive"),
    ("PayPal is still the best way for us.", "dismissive"),
    # [REAL] Aggressive client, image 1
    ("I'm paying you 4.25 rn.", "dismissive"),
    # [REAL] Facebook Marketplace, image 4
    ("It's only 5500 btus.", "dismissive"),
    # [SYNTH]
    ("Whatever, just send the invoice and let's move on.", "dismissive"),
    ("I don't really care about the details, just get it done.", "dismissive"),
    ("Not really interested in the breakdown, just give me a total.", "dismissive"),
    ("I don't have the bandwidth to keep negotiating this back and forth.", "dismissive"),
    ("Doesn't really matter to me, do whatever's easiest for you.", "dismissive"),
    ("Sure, fine, doesn't matter either way, let's just proceed.", "dismissive"),
    ("Not super invested in this conversation honestly, just wrap it up.", "dismissive"),
    ("I'm not going to spend more time discussing this point.", "dismissive"),

    # ------------------------------------------------------------------ #
    # ENTHUSIASTIC
    # ------------------------------------------------------------------ #
    # [SYNTH] — no strong real examples in the batch, all synthetic
    ("This is exactly what we've been looking for, love it!", "enthusiastic"),
    ("Your portfolio is amazing, I'm really excited to work with you on this.", "enthusiastic"),
    ("Yes! Let's do this, I'm genuinely thrilled about the direction.", "enthusiastic"),
    ("This proposal is fantastic, can we start right away?", "enthusiastic"),
    ("I'm so glad we found you, this is going to be great.", "enthusiastic"),
    ("Honestly this exceeded my expectations, awesome work.", "enthusiastic"),
    ("Can't wait to get started, this looks incredible.", "enthusiastic"),
    ("We love your work, genuinely excited about this whole partnership.", "enthusiastic"),
    ("This is perfect, exactly the vision we had in mind.", "enthusiastic"),
    ("I'm really impressed, let's move forward as quickly as possible.", "enthusiastic"),
    ("You're clearly the right person for this, I don't even need to look further.", "enthusiastic"),
    ("This is better than what I imagined, I'm fully on board.", "enthusiastic"),

    # ------------------------------------------------------------------ #
    # NONCOMMITTAL
    # ------------------------------------------------------------------ #
    # [REAL] Girish early in negotiation
    ("Maybe think of an average monthly salary for the next 3 months. Then we review and revise in 3 months?", "noncommittal"),
    ("You suggest something and let me know.", "noncommittal"),
    # [SYNTH]
    ("We'll see, I'll get back to you at some point.", "noncommittal"),
    ("Let me think about it and follow up with you later.", "noncommittal"),
    ("Not sure yet, still weighing a few options before deciding.", "noncommittal"),
    ("I'll loop in the team and let you know eventually.", "noncommittal"),
    ("Can't commit to anything concrete right now.", "noncommittal"),
    ("We're still exploring a few directions before we make a call.", "noncommittal"),
    ("I'll keep this in mind for later, no promises though.", "noncommittal"),
    ("Let's revisit this conversation down the line when things are clearer.", "noncommittal"),
    ("Still gathering info before we make any decisions on this.", "noncommittal"),
    ("I'll let you know if anything changes on our end.", "noncommittal"),

]

REAL_INDICES = {
    # indices of SEED_DATA entries that are [REAL] (not synthetic)
    # used for weighted sampling or analysis
    0, 13, 20, 21, 22, 23, 24, 25, 26,  # hesitant, urgent, aggressive
    35, 36, 37, 38, 39,                  # bluffing (Girish salary comparisons + marketplace)
    48, 49, 50, 51,                      # collaborative (Girish)
    60, 61, 62, 63, 68,                  # dismissive (Girish + aggressive client + marketplace)
    87, 88,                              # noncommittal (Girish)
}


if __name__ == "__main__":
    from collections import Counter
    counts = Counter(label for _, label in SEED_DATA)
    real_count = len(REAL_INDICES)
    print(f"Total examples:    {len(SEED_DATA)}")
    print(f"Real examples:     {real_count}")
    print(f"Synthetic:         {len(SEED_DATA) - real_count}")
    print()
    for label in LABELS:
        print(f"  {label:15s} {counts[label]}")
