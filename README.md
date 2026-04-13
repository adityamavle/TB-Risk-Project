# RCT Simulations for Enhanced TB Treatment Outcomes
This project is a FastAPI-based application designed to simulate datasets based on various population parameters. It offers endpoints for dataset simulation, statistical power calculation, optimization of sample sizes, and generation of clinical datasets.

## Clinical Trial Context
This simulation study is tied to the clinical trial work on identifying biomarkers that can predict progression from latent tuberculosis infection to active tuberculosis disease.

- **Clinical Trial Reference**: http://jhubi-ctu.jhu.edu/research/clinical-trials/identification-of-biomarkers-that-can-predict-progression-from-latent-tuberculosis-infection-to
- **Study Location**: India
- **Topic**: Tuberculosis
- **Enrollment**: Open
- **Trial Period**: Ongoing

The referenced clinical trial builds on the C-TRIUMPH household contacts cohort, including household contacts who progressed from latent TB infection to active TB. Its purpose is to compare immune and biomarker profiles between progressors and non-progressors, including plasma biomarkers, microRNAs, TB-antigen and mitogen stimulated cytokine and chemokine responses, growth factors, and other immune factors.

Study aims include:
- Identify biomarkers that can accurately predict progression from latent TB infection to active TB disease.
- Measure baseline plasma biomarker profiles in household contacts who progress versus those who do not.
- Profile baseline microRNAs in household contacts who progress versus those who do not.
- Measure baseline unstimulated, TB-antigen stimulated, and mitogen stimulated cytokine and chemokine responses.
- Measure baseline unstimulated, TB-antigen stimulated, and mitogen stimulated growth factors and other immune factors.

## Features
- **Simulate Dataset**: Generate datasets based on user-defined population parameters.
- **Calculate Statistical Power**: Compute the statistical power for different treatment groups.
- **Optimize Sample Size**: Adjust sample size to achieve desired statistical power.
- **Generate Clinical Dataset**: Create a clinical dataset with an equal number of samples from each population group.

## App Links
- **Parallel Trial Design**: https://tb-project-web-app.onrender.com/docs
- **N factorial design**: https://api-tb-risk-project.onrender.com/docs
- Local Run Command: uvicorn app:app --reload -> go to /docs

