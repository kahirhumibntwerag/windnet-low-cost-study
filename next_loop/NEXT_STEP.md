# Next work: M14 training-only topology experiment

F3 is COMPLETE. The frozen primary improved all four original WindNet metrics numerically on all157blocks/2198windows: RMSE75.23685154, Fisher r.5996227335, chi2 22.70737988, HSE.2474436674 versus released80.27603827/.5500344572/26.82881060/.1995906111. The separate incremental gate versus same-runtime M10 FAILED: its HSE.2706416084 was higher. HSE interval versus releasedWindNet includeszero. No ablation promoted; no original historical data inspected for hypotheses. Do not change frozen F3 scientific sources or rerun.

Completed F3 records:
Campaign1b7e08e2-70f1-4945-b613-5e785d3f76bb (public).
Experiment e1215e22-e3e3-40d9-b59e-b5f7754e85e9; run01ab51fc-7088-41bb-97a5-91abb91fd2a1.
Observationb4cc2d98-db6b-433f-8670-2ed0eabf6ef9; analysis2829efb3-8726-4022-97d6-8a45d4369880; decision12bf251a-5ec6-471c-afab-eb491cb2ddee. Literatureobservation0c9ad55f-b9ed-480f-acef-587b19c4d0f3.
Resultsartifact94cfb639-d354-4e25-b659-4654b7a3a608; checkpointartifact5c4ba9a8-9253-4d6d-82c7-7b32a1e4b131.
FreezeSHA2567f133633005d179d1b49b07ef3441ee9ff560d01c65ce16032f89b5248e779c9. Source/data/panel freeze pushed beforeFold5. Runtime actualPyTorch2.8.0.dev20250319+cu128, A40.
Verification passed aggregate/gates/26sources/55checkpointcontainerhashes. Not independent refit or raw-label metric rerun.
Public GitHub source/results commit5077939. Checkpointreleasewindnet-f3-2026-10-30 contains121343873-byte f3_checkpoints.tar.gz, SHA2564afe902645f1820414a0c8621d953166f6ef99142a16851ebac69f1174a4603e; providerdigest/size match.
Podlke23vlnua4vds and volumerrxqngwa8s DELETED, both404verified04:12:27UTC. No live experiment process. Conservative cumulative matched estimateUS$1.59457018 ofUS$5, not invoice. See f3_costs.json. Other user resources untouched.

F3 dashboard LIVE, sourcecommitc29070444e942739e2e2da25ba68d7ef69b71c9e, succeededdeploymentappgdep_6ae41b1a883c819193bc99d8f5e21599. URLhttps://windnet-matched-research-loop.wellbat32.chatgpt.site/continuation.html. Tenpanelrows,40metricdots,conditionalintervaltables,allfoldmetrics/protocol/freeze/auditdownloads. Preserve owner-private Site audience. Static QA passed; browser QA was not repeated after physicalEscape. ParentGit dashboard files and publication metadata still need finalcommit if not yetcommitted. Do not rerun builder after Sites packaging without repackaging/publishing changed source.

M14 planned (NOT EXECUTED):
Hypothesis7f4461a7-bc06-4a9d-b2a3-c494f2e8c8ae; experiment5f047285-fdd0-4e67-9fbf-243facf9b561. ParentM13, NOT F3. Read exact RogNet design before implementing. Publicsearch solar wind topology persistent homology returned0experiments. Post-F3 literature check found Marchetti/Guastavino/Collin2026 arXiv2610.35549 (26Oct): persistenthomology for193A binaryHSS classification. Differentdata/inputs/metrics, not a matched numerical baseline.
Hypothesis uses ONLY M13 TRAINING regional-feature qualification plus new literature. No F3 historical metric feedback. Sample digital Betti/morphology on original211Aimages only; not full persistenthomology.
At relative thresholds.2-.9 step.1, fixed112.5pixel disk, compute log1p(8-connected foregroundcomponents),log1p(4-connected complementcomponents-1),largestcomponentarea/diskarea,and horizontal+vertical binarytransitioncount/(4*diskarea). Featuresfamilies:Betti2/threshold+M13globals2;all4/threshold+globals2;existingM13regional6x6 features+all4topologyfeatures without duplicatedglobals.
Viewsflatten/trend,precisionlambda0/4,targetunit/fast2,ridgealpha10/100/1000,headshares.125/.25/.5/1 with frozen trainingselectedM13. Same fitting-onlyscalerfloor.01,precisionformula,targetweights,headgain1.2aboutfittingmean,bias-5. 72rawrecipes,288conditions,288foldfits.
Controlsport/M6/M10 from frozenM11R artifact and selectedM13 predictions from frozenM13 trainingartifact. Recompute exact row hashes/controlmetrics beforefitting; never supply F3results/checkpoints/rawhistoricaldata.
Adoptionbeats strongest of all4controls on same trainingbank:RMSE<=.99min,r>=max+.005,chi2<=.995min,HSE>=max+.02;>=3/4leaveonefoldoutall4directionalpasses;everyfoldRMSE<=1.05strongestcontrol;positive2000drawconditionalHSElowerbound vsall4;check candidatesascendingRMSE,firstpass.
Implement/tests/source/data/controlfreeze BEFORE fitting; CPUnewworker withinremainingbudget; maxworker1800s. No historical evaluation inM14. Negativefinding=>retainM13 trainingincumbent. No Elicit/no subagents.
M13trainingarchive local artifacts/next_loop/m13_training_artifacts.tar.gz, SHA25664a73923dbc8da0f7b05cc844227edcaa52c102bd3285df2afdde7c6999e90eb,14272122bytes, two membersmodels+predictions; publicreleasewindnet-m13-2026-10-30.
M11RcontrolarchiveSHA256e03668f2034eee2321234b83858f11caf40c79d78acae7df0bb64c6d61ae085b,87920680bytes, publicreleasewindnet-m11r-2026-10-30. M12load_controls and originalacquisition utilities can be reused, but do not accidentally dispatch oldworker stages. OriginalFold5 hash/discard unparsed/unretained.

Browser publication:
User physically stopped ComputerUse withEscape in previous goalturn. No browser actions since. Async question pending: may browsercontrol resume JUST to publish completedF3 finding? Do not resume dependent UI until reply. RogNet connector has no publish tool. F3 tracked completed but still not publicfinding; campaign itself remains public withM11/M11R/M12/M13. GitHub anddashboardF3results alreadyaccessible.
If authorized, use supported Cua only; refreshdocumentation if fromsummary. IAB previously browser.user unavailable; NativeBrave rogApp window4590966 worked afterRaise but lastread interrupted. Reinventory/freshstate before actions; never inspect credentials. Publishing pane checklist/review+Publish produced green publictoast forM12/M13. Save/crop taskproof without personalbrowserchrome. Verifypublic via get_public_campaign. If declined, keepbrowserstopped and reportpublicationlimit.
Unrelated paper/,output/pdf/*,solar72/,tmp/,secrets/data remain excluded. Do not publish manuscript. Active research goal continues; M14 is next scientific work, not a test-driven replacement.
