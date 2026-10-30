# Current continuation state

The goal remains active: improve all four original WindNet metrics; choose hypotheses using training-side validation only; search primary literature after each experiment; record everything in public RogNet, GitHub and the existing dashboard. Do not use Elicit or inspect original held-out images, labels, forecasts or residuals for hypotheses.

## Completed research

Public campaign: 1b7e08e2-70f1-4945-b613-5e785d3f76bb. M11 technical failure and M11R negative result remain unchanged. M12 and M13 are published.

M12 experiment97b0bd0d-8bc1-46f7-882c-7d9859ae4bcd completed288conditions:13numerical all-four passes,9fold-screen passes,all9event intervals againstM10crosszero. No adoption. Initial Windows/Linux manifest path error was repaired before fitting, preserving hashes and recipes.

M13 experiment6e39e7e9-7997-494f-beac-57bbe441a006,run c62db121-da88-48d1-ac71-b1b0e5cf0c2d completed960conditions:50numericalpasses,40foldscreenpasses. The24thcandidate in registered RMSE/interval order qualified; first23failed. Later candidates were not declared failures. All192overlapping unit-weight conditions reproduce M12 predictions exactly(maxdifference0).

Training-qualified primary: M13_g6x6_trend_linear_w4_a100_targetfast2_share0.25.

Recipe:6x6image-plane median-relative darkness grid; four-image trend view(last,mean,last-first); linearfeatures; StandardScaler fit-only withscale floor.01; ridgealpha100; precisionlambda4;2x fitting-target weight at y>=500; gain1.2about fitting-target mean,bias-5. Output25%calibrated regionalridge+75%fixed M10recipe. M10 remains ratio/flatten locationridge lambda4 alpha10000,location.625+M6.375,gain1.2,bias-5. M6 keeps equal port/combinedridge mixture and gain1.2.

Training candidate versus fixed matchedM10:RMSE77.09794756 vs77.97714841;Fisher r.5891855691 vs.5730469304;chi2 24.85847072 vs26.88077460;originalHSE.2952163257 vs.2389044289. It passes4/4leave-one-fold-outchecks and has conditionalHSEdelta95vsM10[.003740277,.115285914],also positivevsport/M6. These are adaptive reused-bank training results,not historical performance.

Read results/next_loop/REGIONAL_RESULTS.md, both full results and verification files, and next_loop/m13_training_selection.json. Training recipe is locked; full F3 evaluator freeze is NOT complete. No new original historical evaluation has run.

## Next required work: F3

F3 hypothesisc67e60cb-d36e-4a1e-9c05-0a89b84e9050;planned experimente1215e22-e3e3-40d9-b59e-b5f7754e85e9. Read its exact registered design from RogNet.

Implement/test/hash the full F3 evaluator, data identities and all ten panel recipes BEFORE acquiring originalFold5. Panel:primary,matchedport,M6,M10,regional-only,regionaltargetunit,regionallambda0,regional1x1globalgrid,no outer amplitude calibration,releasedWindNet. Exact overrides/calibration definitions are in RogNet. All five original folds and2198evaluationwindows must match for every model; each fits the other four folds. Use the same current final runtime for every trainable model; do not assert exact numerical reproduction of earlier archived controls. Fit all before exporting scores; prohibit post-test replacement and subsequent hypotheses based on test feedback. Report separate all-four gates and conditional intervals versus releasedWindNet and M10.

Useful immutable code:event_loop/final.py already implements F2 source/data freeze, delayed Fold5 acquisition, label boundaries, all-fits-before-scores, fixed-panel metrics and checkpoints. Reuse helpers in a separate F3 evaluator, without rewriting frozen matched/event sources. Its predict helper caches port/prior/location/affine_location_mix. Add a separate regional ridge/mix prediction path using regional_features.py and target_weights.py. Include all new code in the final freeze. The registered F3 is planned, not started.

AfterM13 primary-literature search reviewed Cawley&Talbot2010JMLR on selection overfitting and originalWindNet arXiv2006.05825. This motivates locking before evaluation, not claiming freshblindness. Prior historical exposure and sequential testing remain limitations.

## Resources and publication

No experiment process is live. M12/M13 pod b2vo3i0njc9ax1 and volume vdexoc96gq are deleted,404verified. Do not reuse their worker URLs/caches. Fold5 was never parsed or retained during development.

Latest cumulative matched cost estimateUS$1.43694941 ofUS$5,from results/next_loop/m12_m13_costs.json. A new F3 worker must stay within the remaining envelope and be separately tracked/cleaned up. Other account resources are not owned by this campaign.

Training archives are local under artifacts/next_loop and public releases windnet-m12-2026-10-30 and windnet-m13-2026-10-30. M13SHA25664a73923dbc8da0f7b05cc844227edcaa52c102bd3285df2afdde7c6999e90eb. M11R control archive remains in release windnet-m11r-2026-10-30.

Dashboard generator:python -m next_loop.build_regional_dashboard. Existing Siteprojectappgprj_6ae3988ae03881919056b61092bec1f4,checkoutmatched_dashboard/,owner-private audience retained. Branchcodex/windnet-matched-loop. Manuscript and unrelatedsolar72work were not changed/published.

Browser note:in-app APIs returned browser.user is unavailable. Existing nativeBrave RogNetwindow worked through cua.getApp({windowId:4590966}) after its Raise action. Publishing is absent from RogNet MCP; the service UI published M12/M13, with confirmation toast. Prefer tools; do not inspect browser/session credentials. Public status can be verified with get_public_campaign.
