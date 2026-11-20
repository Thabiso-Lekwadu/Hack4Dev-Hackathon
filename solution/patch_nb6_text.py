"""Align Notebook 6's markdown with its executed result: Madrid CV selected alpha = 0 (no smoothing)."""
import json, sys
R = [
 ("complementary learners that are averaged: (i)", "complementary learners that are averaged: (i)"),
 ("whose weight fades as more labels arrive. Finally the\nprobabilities are **smoothed over neighbouring buildings**, since construction eras cluster spatially.",
  "whose weight fades as more labels arrive. Spatial smoothing of the predicted probabilities was also evaluated, but\nMadrid cross-validation rejected it (Section 6.4): the spatial *features* already carry the neighbourhood context."),
 ("The average is then smoothed spatially with the α, r chosen on Madrid. The support pixels' labels are known, so the support\nset is included when smoothing — this is legitimate (they are the training labels) and helps neighbouring pixels.",
  "The function can smooth the average spatially with the α, r chosen on Madrid. Madrid CV selected **α = 0**, so the final\nprocedure applies **no smoothing**; the option is kept in the code for other cities and is documented in Section 6.4."),
 ("plus spatial smoothing of predictions. The table above gives the 5 × 5 CV score against Notebook 4's 0.618.",
  "scored with 5 × 5 repeated CV: **0.783 ± 0.004** macro F1 against Notebook 4's 0.618. Prediction smoothing was tested and rejected on Madrid."),
 ("stacker and fading prototypes, then smooths spatially. It improves",
  "stacker and fading prototypes. It improves"),
 ("> estimate of the smoothing benefit.",
  "> estimate of the smoothing benefit.\n>\n> **Result:** every setting *lowered* Madrid CV (α = 0.3, r = 3: −0.002; α = 0.5, r = 3: −0.013; α = 0.7, r = 6: −0.049),\n> so **α = 0 (no smoothing)** is selected and frozen. The spatial features of Notebook 5 already give each pixel its\n> neighbourhood context, so averaging neighbours' predictions as well mostly blurs real boundaries between estates."),
]
def fix(s):
    for a, b in R:
        s = s.replace(a, b)
    return s
nb = json.load(open(sys.argv[1]))
n = 0
for c in nb['cells']:
    if c['cell_type'] == 'markdown':
        src = ''.join(c['source']); new = fix(src)
        if new != src: c['source'] = new; n += 1
json.dump(nb, open(sys.argv[1], 'w'), indent=1, ensure_ascii=False)
b = open('solution/build_nb6.py').read(); open('solution/build_nb6.py', 'w').write(fix(b))
print('patched markdown cells:', n)
