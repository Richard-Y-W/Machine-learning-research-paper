# Follow-up decision and dense-control results

The evidence supports a narrower interpretation of weight access: pretrained parameters can provide useful initialization for realizing an intervention in a fixed network. The fresh-versus-warm contrast does not establish semantic programming of weights. Structured query teachers, target objectives, and synthetic supervision remain essential baselines. A claim about language-model weight programming would require a language-model editor and capabilities beyond these numerical procedures.

Under the fixed long L-BFGS schedule with symbolic targets, original representation, and 128 queries, warm initialization achieved mean physical RMSE 0.0195, compared with 0.0377 for a fresh student. The predeclared warm-minus-fresh contrast was -0.0182 [-0.02255, -0.01401]. Families are averaged within each of the 12 new seed clusters. This comparison isolates initialization under this optimizer and sparse-label schedule. It establishes neither semantic interpretation of weights nor superiority over all query procedures.

With 4,096 free synthetic labels and 2,400 minibatch Adam steps, fresh and warm students achieved physical RMSE 0.0175 and 0.0119, respectively. The paired warm-minus-fresh contrast was -0.005627 [-0.006614, -0.00455]. Both use only the same 128 base-model queries. The dense schedules are a separate cost curve; their optimizer, number of updates, and input distribution differ from sparse L-BFGS. Cross-curve differences cannot be attributed to input density alone. Within each dense schedule, initialization is the only changed factor.

## Dense symbolic distillation

| Student | Adam steps | Physical RMSE | Teacher-fit RMSE | Construction seconds |
|---|---:|---:|---:|---:|
| Fresh dense | 600 | 0.0372 | 0.0369 | 0.8828 |
| Warm dense | 600 | 0.0159 | 0.0150 | 0.7631 |
| Fresh dense | 2400 | 0.0175 | 0.0167 | 3.1218 |
| Warm dense | 2400 | 0.0119 | 0.0108 | 2.9477 |

The primary sparse-supervision contrast was frozen before held-out evaluation. The separately committed dense control strengthens the query-only comparator. Neither comparison is a language-model evaluation.
