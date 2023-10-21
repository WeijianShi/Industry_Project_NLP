# DG Venture Law Firm 

## Instruction Manual--NDA Processing Tool

Welcome to the NDA Processing Tool! This tool is designed by Weijian Shi, Shaoxiong Jia, and Yichen Li. The purpose is
to help you efficiently process Non-Disclosure Agreement (NDA) documents. Please follow
the instructions below for a seamless experience.

### Why This Tool?
In the NDA (None Disclosure Agreement), it always has some dealbreaking words. In different clauses, certain words could be treated
as "deal-breaking". For instance, in the "Disclosure" section, the word "exclusively" is not appropriate,
since in the current stage, both party in the agreement is only signing a non-disclosure agreement, rather than an 
official cooperation contract. Our end-to-end solution aims to detect these errors and provide modification suggestions
to the user. Check out the pipeline's flow chart and instructions below!




### NDA Pipeline Visualization
Here is the workflow of the pipeline
<img src="https://github.com/WeijianShi/Industry_Project_NLP/blob/main/Image/Flowchart.jpeg">
### First Use

1. Click on the three dots located to the right of the folder.
2. Select "Move to" and choose "My Drive."

### Operation Process

3. Upload your Docx file and rename it as "NDA.docx." Make sure to delete the existing "NDA.docx" file if present.
4. Choose between two available options:
   - Option 1: **Only Dealbreaker NDA Specialized Search.ipynb**
   - Option 2: **Only Dealbreaker NDA Global Search.ipynb**
5. Open the corresponding `.ipynb` file.
6. Click on "Runtime" in the menu bar and select "Run all" (or simply press Ctrl + F9).
7. Verify at Google’s double-confirmation by clicking "Yes" or "Confirm" twice.
8. Wait for the below running bar to show "all complete," then close the page directly.
9. You will now see a new docx file named "NDA_db_result.docx."
10. Download this docx file to your local machine.
11. Open the local file to access the desired content. If you are not satisfied with the result, you can try a different method starting from step 1.

### Some Noteworthy Matters

- **Do not modify the code:** Please refrain from making any changes to the provided code.
- **Deleting the original NDA.docx:** After successfully downloading the new docx file, remember to delete the original "NDA.docx" to avoid confusion.
- **Technical Support:** If you encounter any issues or accidentally delete important files, please contact our technical team at ws2679@columbia.edu for assistance.

### Model Selection

- **Only Dealbreaker NDA Specialized Search.ipynb:** This option uses Specialized Search Logic to process NDA documents.
- **Only Dealbreaker NDA Global Search.ipynb:** This option uses Global Search Logic to process NDA documents.

### Pending Search Methods

In the "Pending Search Method" folder, you will find two additional algorithms that incorporate the use of synonyms:

- **Pending: Dealbreakers and Synonyms NDA Specialized Search.ipynb**
- **Pending: Dealbreakers and Synonyms NDA Global Search.ipynb**

Please note that the technical use of these two methods is still under negotiation, and they have been placed in the pending folder.

Enjoy using the NDA Processing Tool, and feel free to reach out to our technical team if you need assistance or have any questions!









