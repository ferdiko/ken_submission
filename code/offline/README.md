# Run offline phase

## main.py

RL & profiling has been run and is written to file

## RL

Run RL and in the end write pareto-frontier to file

## Placement

Do this pruning with the simulator (oder soll ich ohne simulator machen?).
 - [ ] Check how predictable it is etc. check different model placements, you'll need that anyways for algo design and debugging and intuition etc

## Batch sizes

- [ ] How exactly to do the gradient descent
- [ ] Make an automated profiling script. In each iter, profile the ones next to them.
- [ ] Let's first profile all, the thing is it depends on the QPS and burstiness etc so not a good idea?
- [ ] 

## TODOs

END-TO-END SIMULATED EXPERIMENT:

 - [ ] Make bigger dataset
   - [ ] Tune this a bit
 - [ ] Static baseline
   - [ ] Get acc in simulator
 - [ ] Dynamic
   - [ ] Have sim method for dynamic stuff (e.g. input qps array)
   - [ ] 



SUNDAY:

Set up end-to-end experiment  ...

 - [ ] Implement latency SLO
 - [ ] Find a scale factor and latency SLO that's interesting
 - [ ] Can we do that unsimulated

OLD STUFF:

 - [X] Profile workload
 - [ ] Profile models
   - [ ] Model memory foot print
   - [ ] Model inference time
   - [ ] Change GPU capacity to 1s
 - [ ] Hook batch size back in
 - [ ] Make offline run end-to-end. Try to get some results here
 - [ ] Make workload runner with scale factor


 - [ ] Check if there's off by one bug with qps starting from 0

 - [ ] Refactor RL code with the new structure
 - [ ] Build simulator for batch sizes