## OPTION 1: VERCEL (YOU HAVE THIS!)
1. **Run** the `PREPARE_DEPLOY.bat` file in this folder.
2. **Go to:** [Your Vercel Dashboard](https://vercel.com/luminark/luminark)
3. **Connect** your GitHub repository.
4. **Deploy** the `LUMINARK` project.
   - Root Directory: `./`
   - Framework Preset: `Other` (or None)
   - Build Command: `(leave empty)`

## OPTION 2: NETLIFY DROP (FASTEST)
1. **Run** the `PREPARE_DEPLOY.bat` file.
2. **Go to:** https://app.netlify.com/drop
3. **Drag** the entire `LUMINARK` folder onto the page.
4. **Done!**

---

## WHAT TO TEXT NIKKI:

```
Hey Nikki! 

Test this SAP Antikythera Engine I built.
It's the new V6.2 Professional Build.

[YOUR LINK HERE]

Let me know if the "Vectors" section makes sense on your phone.
```

**STATUS:** READY TO DEPLOY 🚀
                    <button class="info-btn" onclick="toggleHelp('h-vector')">[?] Info</button>
                </div>
                <div id="h-vector" class="help-text">Choose the primary container where your energy is most focused.
                    This determines the specific tactical protocols you'll receive (e.g., Career-specific vs.
                    Relationship-specific steps).</div>
                    This determines the specific tactical protocols you'll receive.</div>

                <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 12px; font-weight: 500;">What focus is
                <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 15px; font-weight: 500;">What focus is
                    currently most dominant in your life?</div>
                <div class="vox-options">

                <!-- TIER 1: FUNDAMENTALS -->
                <div style="font-size: 0.6rem; letter-spacing: 2px; opacity: 0.4; margin-bottom: 8px;">TIER 1: CORE
                    FUNDAMENTALS</div>
                <div class="vox-options" style="margin-bottom: 20px;">
                    <div class="vox-option" onclick="selectVector(this, 'Relationship')">Relationship</div>
                    <div class="vox-option" onclick="selectVector(this, 'Career')">Career</div>
                    <div class="vox-option" onclick="selectVector(this, 'Family')">Family</div>
                    <div class="vox-option" onclick="selectVector(this, 'Finances')">Finances</div>
                    <div class="vox-option" onclick="selectVector(this, 'Health')">Health</div>
                    <div class="vox-option" onclick="selectVector(this, 'Growth')">Self-Growth</div>
                </div>

                <!-- TIER 2: MEANING -->
                <div style="font-size: 0.6rem; letter-spacing: 2px; opacity: 0.4; margin-bottom: 8px;">TIER 2: MEANING &
                    PURPOSE</div>
                <div class="vox-options" style="margin-bottom: 20px;">
                    <div class="vox-option" onclick="selectVector(this, 'Spirituality')">Spirituality</div>
                    <div class="vox-option" onclick="selectVector(this, 'Creative Expression')">Creative Expression
                    </div>
                    <div class="vox-option" onclick="selectVector(this, 'Purpose/Mission')">Purpose/Mission</div>
                    <div class="vox-option" onclick="selectVector(this, 'Social Impact')">Social Impact</div>
                    <div class="vox-option" onclick="selectVector(this, 'Legacy')">Legacy</div>
                </div>

                <!-- TIER 3: AUTONOMY -->
                <div style="font-size: 0.6rem; letter-spacing: 2px; opacity: 0.4; margin-bottom: 8px;">TIER 3: FREEDOM &
                    AUTONOMY</div>
                <div class="vox-options" style="margin-bottom: 20px;">
                    <div class="vox-option" onclick="selectVector(this, 'Adventure/Freedom')">Adventure/Freedom</div>
                    <div class="vox-option" onclick="selectVector(this, 'Independence')">Independence</div>
                    <div class="vox-option" onclick="selectVector(this, 'Power/Influence')">Power/Influence</div>
                </div>

                <!-- TIER 4: BELONGING -->
                <div style="font-size: 0.6rem; letter-spacing: 2px; opacity: 0.4; margin-bottom: 8px;">TIER 4: BELONGING
                    & SAFETY</div>
                <div class="vox-options" style="margin-bottom: 20px;">
                    <div class="vox-option" onclick="selectVector(this, 'Community')">Community</div>
                    <div class="vox-option" onclick="selectVector(this, 'Security/Stability')">Security/Stability</div>
                </div>

                <!-- TIER 5: EXPERIENCE -->
                <div style="font-size: 0.6rem; letter-spacing: 2px; opacity: 0.4; margin-bottom: 8px;">TIER 5:
                    EXPERIENCE & RECOVERY</div>
                <div class="vox-options">
                    <div class="vox-option" onclick="selectVector(this, 'Pleasure/Joy')">Pleasure/Joy</div>
                    <div class="vox-option" onclick="selectVector(this, 'Education/Learning')">Education/Learning</div>
                    <div class="vox-option" onclick="selectVector(this, 'Recovery/Healing')">Recovery/Healing</div>
                    <div class="vox-option" onclick="selectVector(this, 'Survival/Crisis')">Survival/Crisis</div>
                </div>
                <input type="hidden" id="dominant_area" value="">
            </div>
Collapse commentComment on line R719foreverforward760-crypto commented on Jan 27, 2026 foreverforward760-cryptoon Jan 27, 2026AuthorMore actionsplease fix codeReactWrite a replyCode has comments. Press enter to view.
            <div class="vector-field">
                <div class="vox-label-row">
                    <div class="label-box" style="margin-bottom: 0;"><span>1. SYSTEM COMPLEXITY</span><span
                            class="num-val" id="c-val">5.0</span></div>
                    <button class="info-btn" onclick="toggleHelp('h-c')">[?] Info</button>
                </div>
                <div id="h-c" class="help-text">Measures the number of moving parts or variables in your current
                    situation. High complexity means many interdependent factors; Low means a simple, focused challenge.
                </div>
                <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 12px; font-weight: 500;">How complex is
                    your current situation?</div>
                <input type="range" id="c" min="0" max="10" step="0.1" value="5"
                    oninput="trackMovement('c', this.value)">
            </div>

            <div class="vector-field">
                <div class="vox-label-row">
                    <div class="label-box" style="margin-bottom: 0;"><span>2. FOUNDATIONAL STABILITY</span><span
                            class="num-val" id="s-val">5.0</span></div>
                    <button class="info-btn" onclick="toggleHelp('h-s')">[?] Info</button>
https://vercel.com/luminark/luminark/HaULm6cCHMyDmNaYEMNcqghcdPjb