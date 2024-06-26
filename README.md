# TP_IGR205_Project
Télécom Paris - IGR205 - Pattern Synthesis project

**Students:**
- Tanguy SCHMIDTLIN,
- Haicheng WANG,
- Romain DAVID,
- Enzo GELAS.

**Supervisors:**
- Pooran MEMARI,
- Amal Dev PARAKKAT.

[Link to the research project subject](https://perso.telecom-paristech.fr/parakkat/IGR205/Projects/PMADP.pdf)

---

### Final implementation
The final implementation can be found [here](./final_implementation) and contains code for line segment pattern synthesis on a blue noise stippling support (using Poisson disk sampling). It can be run using the [run.sh](./final_implementation/run.sh) or through the command line ([source](./final_implementation/code/blue_noise_line_stippling.py)). [Code](./final_implementation/code/blue_noise_point_stippling.py) for classical point stippling is available too.

### Explorations
Other explorations can be found [here](./explorations), either as source files or Jupyter Notebooks. The following approaches have been explored:
- [Line segment pattern synthesis on a weighted Voronoi stippling support](./explorations/voronoi_weighted_sampling/code/voronoi_line_stippling.py) (with [code](./explorations/voronoi_weighted_sampling/code/voronoi_point_stippling.py) for classical point stippling too),
- [Segmentation](./explorations/segmentation/Segmentation.ipynb) postprocessing and evaluation,
- [Noise generation](./explorations/noise_generation/Generation.ipynb) inspired by the [Patternshop](https://xchhuang.github.io/patternshop/index.html) article,
- [Gradient](./explorations/gradient/Gradient.ipynb)-only approach, which was our starting point,
- a [depth estimation](./explorations/depth_estimation/Depth_Estimation.ipynb) approach, as an alternative to the gradient for computing a vector field from the input image. It uses the [MiDaS](https://arxiv.org/abs/1907.01341) depth estimation model.