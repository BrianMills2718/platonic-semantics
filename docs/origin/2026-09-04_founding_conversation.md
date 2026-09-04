i have heard that the latent space of llms for english and chinese and aother langauges have th esame strucutre. and that they share a lot of this sturcture with whale language. and then spearately that llms are kind of indepdnently conversing on the saem kind of structure of latent space. can you look into this and adivse

Yes. There is a real and quite interesting scientific story underneath what you heard, but the strongest version — “English, Chinese, whales, and independently trained AIs all discover the same universal latent language” — goes beyond the evidence.

A useful way to think about the evidence is:

Claim	My assessment
Different human languages share semantic structure inside multilingual LLMs	Strong evidence
An LLM often maps different languages into a more language-independent “concept space” internally	Good evidence, model-dependent
Independently trained neural networks develop geometrically similar representations	Real evidence, but partial
Vision models and language models increasingly resemble one another representationally	Intriguing evidence
All sufficiently capable AIs converge on one universal latent representation of reality	Interesting hypothesis, not established
Whale communication has the same latent space as human language/LLMs	Not demonstrated
Whale communication has structural/combinatorial properties that ML can discover similarly to language	Yes
1. English and Chinese really can become closer “inside” an LLM

At the input and output ends, English and Chinese obviously look extremely different.

But internal activations can behave differently. A 2024 ACL study of Llama-2 found a striking three-stage pattern. When given non-English inputs, early layers represent the input, middle layers move toward something resembling a semantic or “concept” space, and later layers convert the representation into the desired output language. In their experiments, that middle representation was actually somewhat English-biased, presumably partly because Llama-2's training corpus was English-heavy.

Even more interestingly, subsequent work across typologically diverse languages found shared representations of grammatical concepts. So Chinese, English, Arabic, etc. aren't simply living in totally separate neural universes. There appear to be common semantic/grammatical dimensions plus language-specific machinery.

A simplified picture might therefore be:

English words → language-specific encoding → concepts/relations → language-specific decoding → English words

and

Chinese words → language-specific encoding → roughly the same conceptual machinery → language-specific decoding → Chinese words

But I would avoid calling the middle layer “English translated into machine language.” It isn't a list of discrete symbols. It is a very high-dimensional distributed representation.

2. The surprising part is that different models can develop similar geometry

This is probably the other thing you've encountered.

Researchers have noticed that separately trained neural networks often organize concepts in ways that can be aligned.

For example, imagine that one model places:

dog near wolf
dog farther from automobile
red near orange
Paris related to France

Another model may use completely different numerical coordinates, but the relationships between the points can look surprisingly similar.

A 2025 study found shared global and local geometry among language-model embeddings and, remarkably, showed that some steering directions could even be transferred between models after aligning their spaces, despite different dimensionalities.

So “the same latent space” is slightly misleading.

It's more like:

different coordinate systems describing partially similar underlying geometry.

Think of two maps of Earth: one Mercator, one globe coordinates. The numbers aren't the same, but New York–Boston is a short relationship, New York–Tokyo a long one, oceans cluster differently from cities, etc.

3. This led to the fascinating “Platonic Representation Hypothesis”

In 2024, Huh, Cheung, Wang and Isola at MIT gave this phenomenon a provocative name: the Platonic Representation Hypothesis.

They observed that as neural models become better — including models trained on different modalities such as text and images — their representational geometries sometimes become more similar. They proposed that sufficiently capable models might be converging toward a common statistical representation of the underlying world.

That's genuinely fascinating.

Suppose reality contains regularities like:

cat → animal → living thing
Paris → city → country → Europe
falling → gravity → acceleration
anger ↔ threat / conflict / arousal

An intelligence trying to predict reality may be forced to discover many of those relationships regardless of whether its input arrives through English, Mandarin, pixels, or something else.

The provocative possibility is therefore:

world structure

↓

English
Chinese
images
audio
etc.

↓

different learning systems

↓

increasingly similar relational geometry

That is much stronger and more interesting than simply saying “LLMs translate languages internally.”

There's even peer-reviewed work finding partial convergence between representations learned by vision models and language models; larger language models tended to have geometries more similar to representations from vision systems.

But “Platonic” should not be mistaken for “proved”

An important 2025 ICML study tested the idea more critically and found that similarity can depend substantially on training objectives and on which datasets you use to measure the representations. In other words, convergence isn't some clean inevitable law that has already been established.

There are several mundane reasons models could resemble one another:

they see overlapping internet data;
they use related Transformer architectures;
they optimize similar next-token or contrastive objectives;
they are evaluated using human-created concept categories;
mathematical alignment methods can deliberately remove differences such as rotations.

So the Platonic hypothesis is currently a very interesting research hypothesis, rather than evidence that we've discovered the unique mathematical language of reality.

4. Now to the whales — this is where I think the story has become exaggerated

Project CETI and collaborators have produced genuinely remarkable results on sperm-whale communication.

Sperm whales communicate using sequences of clicks called codas. In a 2024 Nature Communications paper, researchers analyzing a large corpus discovered contextual and combinatorial structure: features could be varied and combined systematically rather than every coda simply being one indivisible call.

That's significant because combinatoriality is one of the features that makes human linguistic systems powerful.

CETI is now using machine learning explicitly to find such structure and connect vocalizations to social and behavioral context.

And in 2025, researchers introduced WhAM, a Transformer-based generative model of sperm-whale vocalizations. Its learned representations capture useful information about things such as coda rhythm and social units.

But that is not evidence that whale communication shares the same latent space as English and Chinese.

I could not find a credible published result demonstrating something like:

whale concept vector ≈ human-language concept vector

or showing that a transformation aligns whale representations with an LLM's semantic space in the way English and Chinese representations can sometimes be aligned.

That would be an extraordinary result, and we aren't there.

There is nevertheless a deeper connection worth taking seriously

The interesting commonality isn't necessarily language.

It may be structure.

An intelligent system encountering the world has to encode things such as:

identity
similarity
sequence
agency
space
time
social relationships
cause and effect
prediction
hierarchy

Those constraints exist regardless of whether the observer is:

a human speaking Mandarin,

a human speaking English,

a sperm whale communicating acoustically,

a vision model,

or an LLM predicting text.

Consequently, you might expect some representation spaces to have analogous geometry, without expecting their representations to be identical.

And that's the version of the idea that I find scientifically plausible.

There's a lovely distinction here

There are really three hypotheses, and people often collapse them into one:

A. Universal language hypothesis

All intelligent beings ultimately encode thoughts in essentially the same internal language.

Evidence: weak/speculative.

B. Universal conceptual-geometry hypothesis

Because everyone inhabits the same structured reality, sufficiently capable predictive systems will discover some of the same relationships between things.

Evidence: increasingly interesting and empirically testable.

C. Universal mathematical constraints hypothesis

Any system doing compression and prediction will discover recurring abstractions because those abstractions efficiently describe structured data.

Evidence: quite plausible, and it doesn't require anything mystical about language.

I think B + C are where the really interesting science lies.

One very important implication

If this trend proves robust, it may tell us something quite profound about intelligence.

We traditionally imagine languages as fundamentally different representational systems:

English → thoughts
Chinese → thoughts
whale signals → whale thoughts
vision → visual understanding

But perhaps the better picture is:

                         ┌─ English
                         │
                         ├─ Mandarin
REALITY → CONCEPTS ──────┼─ vision
 /RELATIONSHIPS          │
                         ├─ sound
                         │
                         └─ maybe animal communication

The surface codes could be radically different while the relational structure underneath is constrained by the same world.

That would mean AI isn't necessarily discovering a mysterious secret language.

It could simply be discovering regularities of reality that every sufficiently good predictor is forced to represent.

And that interpretation is both less sensational and, to me, considerably more profound.

One phrase worth remembering if you want to explore this literature is “Platonic Representation Hypothesis.” The multilingual work on “latent language” and the Project CETI sperm-whale research are then two adjacent but currently distinct research programs.

If you'd like, I can go one level deeper and explain what researchers actually mean mathematically by two latent spaces having “the same geometry”—e.g. rotations, linear maps, cosine distances, manifolds and representational similarity. That's where this becomes particularly interesting.

Edit
Edit
Edit

k im not sreally sure what i am looking for but maybe this ocnversation about a related proejct of mappingplatonic space in math is relvant 'i want a map of the plaotonic space of mathematical objects. what currently exists as far as this

Yes. What you’re describing is more ambitious than a “map of mathematics.” It’s closer to an atlas of mathematical being: objects as nodes, with edges such as “is a,” “is a special case of,” “is equivalent to,” “maps to,” “classifies,” “is invariant under,” “is dual to,” and “is constructed from.”

There is no accepted, comprehensive map of this Platonic space yet. But several existing projects contain surprisingly large pieces of it.

The closest things that currently exist

1. Formal proof libraries — the strongest skeleton. Lean’s mathlib is probably the most interesting substrate today because definitions, structures, theorems, instances, and dependencies are machine-readable. A 2026 analysis extracted a graph of roughly 308,000 declarations and 8.4 million dependency edges. There are already interactive projects such as Lean Graph and ProofGraph exploring this network. The important limitation is that proof dependency is not identical to conceptual proximity: the network analysis explicitly finds that logical dependency structure can diverge substantially from mathematicians’ conceptual taxonomy.

2. nLab — perhaps the closest human-written conceptual graph. nLab organizes mathematics particularly through category theory, higher category theory, homotopy theory, and type theory. Its strength for your project is that it emphasizes relationships between structures, universal properties, adjunctions, equivalences, and generalizations rather than simply sorting things into disciplines.

3. OntoMathPRO — explicitly an ontology of mathematical objects. This is unusually close to your wording. It represents mathematical concepts using both a hierarchy of mathematical objects and a hierarchy of relationships between those objects. It was designed as Linked Open Data rather than merely as a subject classification. Its scope, however, is nowhere near all of mathematics.

4. OpenMath / OMDoc / MMT — languages for describing the space. OpenMath is explicitly a semantic representation standard for mathematical objects rather than their visual notation. OMDoc/MMT goes further and distinguishes mathematical objects, statements such as definitions and theorems, and whole theories connected by meaning-preserving morphisms. This is important because a true atlas probably needs something like this as its underlying ontology.

5. LMFDB — an actual atlas of a restricted mathematical universe. Within number theory and arithmetic geometry, LMFDB comes surprisingly close to the vision: individual elliptic curves, number fields, modular forms, L-functions, representations, groups, etc. have identities, invariants, permanent pages, and links to related objects. Its explicit purpose includes exhibiting mathematical connections such as those predicted by the Langlands program.

6. OEIS — another enormous region of object-space. OEIS currently contains nearly 399,000 integer sequences. Even more interestingly, a 2026 project called The Map of Integer Sequences extracted 400 mathematical concepts from hundreds of thousands of OEIS entries and generated a network with 19 automatically detected communities. That is essentially an empirical “map of mathematics as seen through integer sequences.”

7. Older “maps of mathematics.” Dave Rusin’s Mathematical Atlas literally provided a clickable territorial “MathMap,” but its organizing units are disciplines and subjects rather than mathematical objects. That distinction is precisely where your idea goes beyond most previous maps.

8. Specialized universes. Wolfram’s Functions Site maps thousands of identities and interrelationships among mathematical functions. Mizar similarly gives a large verified body of definitions and theorems, and researchers have turned the Mizar Mathematical Library into an explicit knowledge graph. These are more pieces of the same puzzle.

Here’s a visual model of how I think these pieces fit into the thing you’re describing:

App block preview

svg

Give feedback

The conceptual jump I think is important is this:

A normal map of mathematics says

algebra → group theory → finite groups → sporadic groups

whereas your map would allow you to descend all the way to

group → finite simple group → sporadic simple group → Monster group

and then move sideways:

Monster → representation theory → modular functions → j****-invariant → elliptic curves → modular forms → L-functions

That sideways motion is where the “Platonic landscape” starts to become genuinely interesting.

And there probably cannot be one ordinary 2-D geometry for it. The space naturally has several different metrics simultaneously:

ontological distance: how many abstraction/specialization steps separate two objects;
constructive distance: how many constructions take one to the other;
categorical distance: morphisms, functors, adjunctions, equivalences and dualities;
logical distance: how much one definition/theorem depends on another;
invariant similarity: objects sharing spectra, homology, cardinality, dimension, symmetry groups, etc.;
computational distance: whether one can be transformed or reduced to another;
surprise distance: objects from apparently remote regions connected by deep theorems — monstrous moonshine and the Langlands program are extreme examples.

So I would not start by drawing continents called Algebra, Geometry, Analysis, etc. Those are partly historical departments in human mathematics. I’d build the underlying thing as a typed, multi-layer hypergraph of mathematical objects, and let the “geography” emerge from whichever relation/metric the viewer selects.

A particularly compelling first prototype could fuse nLab's conceptual links + Mathlib's formal dependency graph + Wikidata/OntoMath-type identity information + OEIS/LMFDB concrete objects. That would get surprisingly close to an explorable first version of the thing you're describing.

The missing piece isn't primarily more mathematical data. It's the universal identity-and-relationship layer that says when two systems are talking about the same mathematical object or construction. OpenMath/MMT and ontology work point toward it, but nobody has yet assembled the full “Google Maps of mathematical reality.”"






Yes. I think that earlier conversation identifies the piece that was missing from what we were doing.

There are actually two different “Platonic spaces” here, and combining them gives you a much stronger project.

1. The two spaces

A. Mathematical-object space

This is an explicit network of things such as:

group → abelian group → cyclic group → Z/5Z

with additional relationships such as:

group --has representation--> vector space

elliptic curve --associated to--> L-function

manifold --has invariant--> homology group

vector space --dual--> dual vector space

This isn't hypothetical. Pieces of this graph exist in Mathlib, OntoMathPRO, OMDoc/MMT, LMFDB, etc. Mathlib alone now has a machine-extracted dependency network of about 308,000 declarations and 8.4 million dependency edges.

B. Neural representation space

This is what the Platonic Representation Hypothesis is talking about.

A model sees things and internally produces a geometry where some stimuli are closer together than others. The remarkable observation is that different models sometimes increasingly agree about those distances, despite having different architectures and coordinate systems.

The project becomes much more interesting when you put those side-by-side.

What I think your actual project could be

Instead of asking:

“Can we visualize the latent space of several LLMs?”

ask:

“Do independently trained intelligences reconstruct the relational structure of mathematics?”

That is an experimentally testable version of the philosophical idea.

You construct an external mathematical graph:

                         mathematical object graph
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
           Group               Circle           Prime number
              │                   │                   │
        ┌─────┴─────┐       ┌────┴────┐          Sophie Germain
        │           │       │         │              prime
   Abelian group  Lie group  Sphere  π₁(S¹)=Z          │
        │           │                    │              │
   Cyclic group     └──── symmetry ──────┘         number theory

Then independently ask several models to represent those exact same objects.

For example:

                         FORMAL MATHEMATICAL SPACE
                                 │
                ┌────────────────┼────────────────┐
                ↓                ↓                ↓
             GPT-like         Qwen-like        Llama-like
                │                │                │
             English          Chinese          English
                │                │                │
                └───────────────┬┘
                                ↓
                     CONSENSUS NEURAL GEOMETRY
                                │
                                ↓
                    compare with formal geometry

Now you have something scientifically meaningful to measure.

Mathematics is an unusually good test case

It solves a major problem with our previous WIT/image experiment.

With ordinary concepts, we don't really know what the “correct” semantic distance between:

dog
democracy
sadness
France
gravity

is supposed to be.

There's no ground-truth geometry.

Mathematics gives us considerably more structure.

For example we know explicitly that:

group
  ↓ is-a
abelian group
  ↓ is-a
cyclic group

and:

topological space
       ↓
fundamental group
       ↓
group

and:

elliptic curve
       ↕
modular form
       ↕
L-function

Formal systems can encode definitions, dependencies, inheritance and morphisms instead of us inventing the relationships after seeing the embeddings.

OntoMathPRO was explicitly designed as an ontology of mathematical concepts and relations. OMDoc similarly focuses on encoding the semantics of mathematical objects, statements and theories rather than merely their written notation.

That's extremely useful for your idea.

And there is a killer experiment hiding here

Suppose we pick 2,000 mathematical objects.

For each object we produce equivalent descriptions in:

English

A group is a set equipped with an associative binary operation...

Chinese

群是一个配备满足结合律的二元运算的集合...

symbols / formal Lean

structure Group ...

diagram

perhaps a Cayley graph or commutative diagram.

Then feed these representations into independently trained systems.

We obtain:

G_English
G_Chinese
G_Formal
G_Model_A
G_Model_B
G_Model_C

where each G is the measured relational geometry.

And separately:

G_Math

is our graph constructed from formal mathematical relations.

Now test:

Does language disappear?

Does

distance(Group, Abelian Group)

look approximately the same when the concepts are presented in Chinese and English?

Does model identity disappear?

Does Qwen independently reconstruct approximately the same relationships as Llama?

Does modality disappear?

Does a model looking at a Cayley graph place “cyclic group” near the same conceptual neighborhood as a language model reading its definition?

Most importantly:
Does the neural consensus approach the formal mathematical graph?

That would be a spectacular result.

Not:

“Models have similar embeddings.”

But:

Different learning systems independently recover measurable portions of an externally specified mathematical relational structure.

That is a much stronger statement.

The map I would build now

And this changes the UI considerably from the scatter plot I built earlier.

I'd make the main view something like:

┌────────────────────────────────────────────────────────────────┐
│  MATHEMATICAL REPRESENTATION ATLAS                             │
│                                                                │
│  Search: [ Elliptic curve                              ]       │
│                                                                │
│   VIEW                                                         │
│   ○ Formal mathematics                                        │
│   ○ Llama                                                     │
│   ○ Qwen                                                      │
│   ○ Claude-like model                                         │
│   ● Consensus                                                  │
│   ○ Difference / residual                                     │
│                                                                │
│                       algebra                                  │
│                         ●                                      │
│                    group ─────────── representation             │
│                     / │                  │                      │
│              abelian  │                module                   │
│                 ●     │                   ●                     │
│                  \    │                  /                      │
│                   cyclic group                                │
│                        ●                                       │
│                                                                │
│              ● topology                                       │
│             /        \                                         │
│       manifold       fundamental group                         │
│          ●───────────────●                                     │
│                                                                │
│                                      ● elliptic curve          │
│                                     / \                        │
│                            modular form   L-function            │
│                                 ●──────────●                    │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│ Relation lens                                                  │
│ [✓] is-a    [✓] equivalent    [ ] proof dependency             │
│ [✓] maps-to [ ] historical    [✓] invariant                    │
└────────────────────────────────────────────────────────────────┘

But the really good feature would be a morph slider:

FORMAL MATH                                          MODEL LATENT

      0% ───────────────────●────────────────────────── 100%
                            63%

As you drag it, objects move from where the formal mathematical graph puts them toward where a particular LLM puts them.

That instantly communicates convergence and disagreement.

Even better: show disagreement, not merely the map

Imagine clicking circle.

The right panel says:

CIRCLE

Formal nearest relations
────────────────────────
sphere
ellipse
S¹
Euclidean space
fundamental group
rotation group


Consensus neural neighbors
──────────────────────────
sphere                   0.94
ellipse                  0.91
disk                     0.88
rotation                 0.82
π                        0.78


Cross-model agreement
████████████████░░░░  82%


LANGUAGE STABILITY

English ↔ Chinese        0.96
English ↔ Arabic         0.91
Chinese ↔ Arabic         0.90


FORMAL ↔ NEURAL
geometry agreement       0.76

Now your visualization isn't decorative.

It's an instrument.

There should probably be multiple geometries

This is another insight from the mathematical-map conversation that I'd keep.

There probably isn't one meaningful distance function.

A pair of objects can be near each other in one sense and very far apart in another.

Consider:

circle ↔ complex numbers

Taxonomically: fairly distant.

But through Euler's formula and rotations: strongly related.

So the UI should offer lenses:

Lens	“Near” means
Ontology	one object generalizes/specializes another
Construction	one can be constructed from the other
Proof	results depend on one another
Categorical	morphisms/functors/duality
Invariant	shared mathematical invariants
Text semantic	mathematicians describe them similarly
LLM latent	neural models represent them similarly
Consensus	relationship survives across models/languages

This is much closer to an atlas than a single embedding plot.

One caution about Mathlib

There's an important result in the new Mathlib network analysis that actually strengthens the case for doing this carefully.

The authors found that formal dependency structure can diverge significantly from mathematicians' human taxonomy, and network centrality frequently reflects formal-language infrastructure rather than mathematical importance.

So:

Mathlib dependency graph ≠ Platonic mathematical space

It's one measurement of it.

Exactly as:

Qwen latent geometry ≠ Platonic space

and:

English descriptions ≠ Platonic space

That's why your project becomes interesting:

Let every source vote on the geometry.
A better architecture

I'd represent the underlying thing as a typed graph:

OBJECT
  │
  ├── is_a ──────────────→ OBJECT
  ├── special_case_of ───→ OBJECT
  ├── equivalent_to ─────→ OBJECT
  ├── dual_to ───────────→ OBJECT
  ├── constructed_from ──→ OBJECT
  ├── has_invariant ─────→ OBJECT
  ├── maps_to ───────────→ OBJECT
  ├── appears_in ────────→ THEORY
  └── used_by ───────────→ THEOREM

LMFDB already follows a useful philosophy at the object level: individual mathematical objects receive meaningful identifiers/pages, invariants and links to related objects.

Then separately attach learned coordinates:

                 OBJECT: cyclic_group

formal_graph_position       [...]
llama_english_position      [...]
llama_chinese_position      [...]
qwen_english_position       [...]
qwen_chinese_position       [...]
vision_position             [...]
consensus_position          [...]

agreement_score             0.87
formal_alignment            0.79
language_invariance         0.94
model_invariance            0.83

That data model can grow indefinitely.

And this is where the whale idea eventually comes back

There is a coherent progression:

                    ┌── English
                    │
                    ├── Chinese
                    │
MATHEMATICAL        ├── formal logic
STRUCTURE ──────────┼── diagrams
                    │
                    ├── different LLMs
                    │
                    └── different architectures
                           │
                           ↓
                    shared geometry?

Mathematics gives you the clean-room test.

If that works, you extend outward:

mathematics
    ↓
physical objects
    ↓
ordinary concepts
    ↓
perception
    ↓
social concepts
    ↓
non-human representations

The whale question then becomes much less woo-woo.

You're no longer asking:

“Do whales secretly speak the same latent language as us?”

You are asking:

“Which relational structures remain invariant when independently learning agents encode the same underlying world?”

That is a credible scientific question.

I would actually change our project direction

Rather than immediately trying to make a giant “map of everything,” I'd make mathematics the first laboratory for Platonic-space mapping.

The resulting project has three layers:

Layer 1 — Mathematical Atlas

Human/formally specified relational structure.

Layer 2 — Neural Atlas

Where different models/languages actually place those mathematical objects.

Layer 3 — Platonic Estimate

The relationships that survive across formal systems, languages, representations and independently trained models.

And critically, we'd visualize uncertainty/disagreement rather than pretending Layer 3 is ground truth.

That feels considerably more rigorous—and considerably more interesting—than the generic cluster map we started with.

If we're going to build an actual artifact next, this is the UI/data model I'd build around rather than polishing the existing scatterplot.