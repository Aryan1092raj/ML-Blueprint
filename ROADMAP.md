# Roadmap

Where this is going. Dates are hopes, not promises.

## Now: setting up

Getting the foundation right before algorithms start landing.

- [x] Folder structure and the shared `core` module
- [x] Writing standards and the algorithm folder template
- [x] Contribution process, issue and pull request templates
- [x] Automated checks on every pull request
- [x] A basic visualiser app
- [] **Three example algorithms**, built by the maintainers: linear regression, k-means, and a
      decision tree

Those three matter more than they look. Every later contribution gets compared to them, so it
is worth getting them right before opening things up. Once they are in, the project is open for
algorithm contributions.

## Next: the classics

Linear and logistic regression, ridge and lasso, k-nearest neighbours, naive Bayes, decision
trees, random forests, gradient boosting, SVMs.

Target is around 15 algorithms done properly. Not rushed.

## After that: unsupervised learning and optimisers

k-means, DBSCAN, hierarchical clustering, Gaussian mixture models, PCA, SVD, t-SNE, and the
gradient descent family up to Adam.

By this point the visualiser should have enough panels to be genuinely useful for teaching.

## Later: neural networks

Building a small automatic differentiation engine from scratch, then layers on top of it.

If we get this one right it becomes the thing people link to when they want to explain how
PyTorch actually works underneath. That is the goal worth aiming at.

## Much later

Once a folder like `tree/` is properly finished and well tested, it could move out into its own
repository and become a small standalone library. That is a long way off, but it is why the
folders are kept separate and tidy now.

## What counts as success

Not stars. Two things:

1. Students at IIT Mandi actually use this to learn, and say it helped.
2. It is still alive after the people who started it have graduated.

The second one is the hard part.
