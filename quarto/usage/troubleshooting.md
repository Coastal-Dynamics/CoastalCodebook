# Troubleshooting

## Github issues
Do you run into an issue with the installation or notebook execution? Don't worry, it's likely that other students have come across similar trouble. To encourage a collaborative learning environment, we strongly encourage you to share your questions or issues by opening an issue on our [GitHub repository](https://github.com/Coastal-Dynamics/CoastalCodebook).

Please note that we prioritize the GitHub issue tracker for all troubleshooting and support. Direct email requests related to notebooks will not be
addressed!

### Why Open an Issue?

- **Collaborative Problem-Solving**: Issues opened on GitHub can be seen, discussed, and resolved collaboratively by the community.
- **Avoid Duplication**: By documenting your problem and the solution on GitHub, you help prevent repeated queries and enable others to benefit from the resolution.
- **Improve the Resource**: Your contributions through issues help us identify and fix bugs, leading to improvements in the course material for everyone.

### How to Open an Issue

If you're unfamiliar with the process of creating an issue on GitHub, [here is a helpful guide](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) that walks you through the steps. It's a simple process that involves:

1. Navigating to the Issues Tab: Click on the 'Issues' tab in the repository.
2. Creating a New Issue: Use the 'New Issue' button to start.
3. Describing the Problem: Provide a clear and concise description of what the issue is,
   including steps to reproduce it, expected outcomes, and any screenshots if applicable.

## JupyterHub users

Normally, your Codebook on the TU Delft JupyterHub should run without any problems. However, if a widget does not load or a question becomes unresponsive, the problem may be caused by leftover or conflicting JupyterLab sessions and kernels.

Before asking for help, try the following rescue steps. They usually resolve the problem.

1. *Save and close all other notebooks.*   Save (**Ctrl + S**) and close (**Alt + W**) all notebooks except the one you are currently working on.

2. *Shut down unused kernels.*
   In the file browser, look for notebooks with an active kernel indicator (a small dot next to the notebook name). Right-click each notebook with an active kernel and select **Shut Down Kernel**. Keep the kernel for your active notebook running.

3. *Restart your active kernel and clear the output.*
   In your active notebook, select **Kernel → Restart Kernel and Clear Outputs of All Cells**.

4. *Run the notebook again.*
   Rerun the notebook cells from the beginning. In most cases, the problem should now be resolved.

5. *If the problem persists, reset the JupyterLab workspace.*
   First, make sure your notebook is saved. Then select **File → Reset JupyterLab Workspace**.
