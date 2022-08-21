# Learning-Repo Post Guidelines
In general, my goal is to disseminate the work that I add to this repository. Most of the learning is self-oriented, but I believe that sharing some of the foundational insights that I gain during these topic explorations may be useful for others.

As of July 2022, the ultimate goal for each of these topics is to create Medium posts. These posts can then be targeted towards publications like TowardsDataScience or TowardsAI.

### TOC
1. [Writing Process](#1writing-process)
    1. [Article Structure](#11-article-structure)
2. [Formatting Guidelines](#2formatting-guidelines)
    1. [Latex - Inline](#21-latex---inline)
    2. [Latex - Display](#22-latex---display)
    3. [Code](#23-code)
    4. [Images](#24-images)
    5. [Videos](#25-videos)
3. [Publication Process](#3publication-process)
    1. [Outline and Resource Medium Formatting](#31-outline-and-resources-medium-formatting)

## 1. Writing Process
Each of the topics is initially drafted as a Jupyter notebook. This draft will primarily contain code, but it can also contain key thoughts & outlines. After the skeleton of the code is completed, I start writing and editing the article on Medium.

### 1.1 Article Structure
The structure of the article should be as follows:
1. Heading
2. Subheading 
3. Header Media
4. Outline
5. Section 1: What is ___?
    - Description of and introduction to the topic. Satellite view. Answer general questions.
6. Sections 2 -> n: Article content
7. Section n+1: Resources

## 2. Formatting Guidelines
This section would be quite short if I wasn't working to publish these notebooks in Medium. I could instead make these posts a `fastpages.fast.ai` blog, but at the moment I prefer the process of creating Medium posts as well as their formatting and search indexing. Because I want to use Medium, I have to deal with the limitations of Medium. This means no automatic Latex and no Python-formatted code.

### 2.1 Latex - Inline
Medium isn't compatible with Latex, but it does support [Unicode characters](https://medium.com/blogging-guide/using-symbols-shapes-and-characters-on-medium-39bc576b9c13). As such, rather than using inline equations, I instead keep inline equations simple and use ***bold-italicized*** equation replacements.

For example, rather than typing $f(x) = x^2$ with Latex, I have to settle for ***f(x) = x<sup>2</sup>***, which is achieved by typing the following in Medium.
> ***f(x) = x^2\*\*\*

**Frequently Used:**
- Superscripting in Medium is achieved with a caret: ^
- Unicode characters must be typed in decimal format in markdown. They can then be imported to the Medium document (see Publication Process)
    - [https://unicodelookup.com](https://unicodelookup.com) for character lookups lookup
    - e.g., &#8721; is created by typing `&#8721;` in the markdown cells.


### 2.2 Latex - Display
Instead of creating latex display blocks, latex displays must be inserted into the post as images (credits to [this guideline](https://medium.com/@tylerneylon/how-to-write-mathematics-on-medium-f89aa45c42a0)).

Each post directory with display equations should have a `\latex\` subdirectory. In the subdir, there should be a `equations.md` file with the latex used to generate the formula.

Each formula should then be converted to an image on [latex.codecogs.com](https://latex.codecogs.com/eqneditor/editor.php).
The images should be saved with the following settings:
![](https://user-images.githubusercontent.com/70919881/181622453-a0cd79c9-d948-4386-a788-76143d2e5599.png)

Then, the images should be uploaded to the `/latex` dir with the following filename image_name**2x**.png where **2x** is required. The images should then be inserted into markdown cells as such:
> `![](/post_dir/latex/image_name@2x.png)`

The images can then be easily pasted into Medium from the .md post.

### 2.3 Code
For pretty formatting, code blocks must be formatted as gists. The gists should be titled with the post name followed by the # of their order of appearance, e.g., `pca_1.png`.

### 2.4 Images
Images should be stored in a `/media/` subdirectory, and should be inserted into markdown cells as such:
> `![](/post_dir/media/image_name.png)`

### 2.5 Videos
Any generated post videos should be uploaded to [https://gfycat.com](https://gfycat.com)], and the link to the gfycat post should be simply pasted where the video belongs. Once the gfycat link is pasted into Medium, it then has to be formatting as a video media line.

### 2.6 Tables
Tables should be converted to csv files and then converted into gists. The links should be inserted into the appropriate markdown location.

## 3 Publication Process
At this stage, the topic exploration should be complete and ready to prepare for a publication. The writing and editing for the article should all take place on Medium.

First, all of the codeblocks should be converted into gists and the gist links should be pasted plainly into the file (see above).

Then, the file should be proofed and grammar checked to prepare an ultimate version.

After an ultimate copy of the markdown file is prepared, the post can then be pasted into Medium quite easily.

### 3.1 Outline and Resources Medium Formatting
The outline at the beginning of the article and the resources at the end of the article should be inserted as code blocks. 

The outline should be hyperlinked to the sections of the post. The post sections can be identified after publishing the post. Open a copy of the published article in Firefox, and press "⌘+⇧+c" to inspect the heading html, and copy the heading "id". The "id" should then be inserted as a hyperlink to the outline table by pasting the id after a pound "#" symbol. This will hyperlink the outline title to the section of the post without causing the post to be reloaded.

The resources should be hyperlinked.

---

Last Update: 08/21/2022
