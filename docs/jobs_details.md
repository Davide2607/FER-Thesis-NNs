# Evaluation Script
920934 -> runs only sbatch (python lines are commented) and without the ld, to see if tensorflow needs that or not
920956 -> has ld not commented and ran both adele and occluded but with only first model 


<!-- Since 922549 didn't run even until running 922570, it's probable that it will have the alterations that were later added for 922554 and 922570 bc of how slurm works. Next time I'll try and use checkouts -->
922549 -> I tried removing "cd /home/dbuhnila/models/FER-Thesis-NNs; pwd"
922554 -> cd is still there now, but gpt5 removed the yolo import and put it lazily inside evaluate() or what it's called, as maybe it was that that changed the paths.
922570 -> I removed cd again and left the lazy yolo just like 922554 (basically a mix of 922549 and 922554)