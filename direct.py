# Sample response from the API
response_text = """
## Eye Analysis

### Abnormalities/Concerns
The most prominent abnormality is the redness in the conjunctiva (the membrane that covers the white part of the eye and inner eyelids). This suggests inflammation. There's also a slight glassy appearance which could indicate irritation or excess tearing.

### Potential Conditions
* **Conjunctivitis (Pink Eye):** This is the most common cause of red eye and can be caused by viruses, bacteria, allergies, or irritants.
* **Blepharitis:** Inflammation of the eyelids, often causing redness, itching, and crusting.
* **Dry Eye Syndrome:** Insufficient lubrication of the eyes, potentially leading to redness, irritation, and a feeling of dryness.
* **Subconjunctival Hemorrhage:** A broken blood vessel in the eye, causing a bright red patch. Although less likely given the diffuse redness in the image, it's still a possibility.
* **Uveitis:** Inflammation of the uvea (the middle layer of the eye), which can be a serious condition requiring urgent attention. However, this typically presents with pain, light sensitivity, and blurred vision, which cannot be determined from the image.
* **Allergic Reaction:** Exposure to allergens like pollen, dust, or pet dander can cause red, itchy eyes.

### Recommendations
It is **crucial** to consult an ophthalmologist or optometrist for an accurate diagnosis and treatment. Self-treating eye conditions can be dangerous. Do not use over-the-counter eye drops without professional guidance. Describe any associated symptoms like itching, burning, pain, discharge, or vision changes to your doctor.

## Oral Cavity Analysis

### Abnormalities/Concerns
There are two noticeable abnormalities at the back of the throat. One appears as a raised, reddish lesion, possibly with a slightly whitish center. The other is a smaller, whitish/yellowish lesion nearby. The surrounding tissue also appears inflamed and red.

### Potential Conditions
* **Tonsillitis:** Inflammation of the tonsils, often caused by a bacterial or viral infection.
* **Strep Throat:** A bacterial infection causing a sore throat, fever, and sometimes white patches on the tonsils.
* **Peritonsillar Abscess:** A collection of pus behind the tonsils, which can be a serious complication of tonsillitis.
* **Oral Thrush (Candidiasis):** A fungal infection causing white patches in the mouth. Less likely given the redness and other lesion.
* **Aphthous Ulcer (Canker Sore):** Although often occurring on the inside of the cheek or lip, they can appear on the soft palate. Less likely given the surrounding inflammation.
* **Viral Infection (e.g., Herpangina, Hand, Foot, and Mouth Disease):** These can cause sores in the mouth and throat.

### Recommendations
A visit to a physician or dentist is **strongly recommended**. They can properly diagnose the lesions and determine the appropriate treatment. Describe any associated symptoms like sore throat, difficulty swallowing, fever, or swollen glands. Delaying treatment for infections in the oral cavity can lead to complications.
"""

# Initialize an empty dictionary with specified keys
sections = {
    'eye_findings': '',
    'eye_conditions': '',
    'eye_recommendations': '',
    'oral_findings': '',
    'oral_conditions': '',
    'oral_recommendations': '',
    'overall_assessment': ''
}

# Function to clean text by removing unnecessary symbols
def clean_text(text):
    # Remove markdown symbols
    cleaned = text.replace('#', '').replace('*', '').replace('**', '').strip()
    return cleaned

# Extract Eye Analysis findings
eye_analysis = response_text.split("## Eye Analysis")[1]
eye_findings = clean_text(eye_analysis.split("### Potential Conditions")[0])
eye_conditions = clean_text(eye_analysis.split("### Potential Conditions")[1].split("### Recommendations")[0])
eye_recommendations = clean_text(eye_analysis.split("### Recommendations")[1])

# Store extracted data into sections dictionary
sections['eye_findings'] = eye_findings
sections['eye_conditions'] = eye_conditions
sections['eye_recommendations'] = eye_recommendations

# Extract Oral Cavity Analysis findings
oral_analysis = response_text.split("## Oral Cavity Analysis")[1]
oral_findings = clean_text(oral_analysis.split("### Potential Conditions")[0])
oral_conditions = clean_text(oral_analysis.split("### Potential Conditions")[1].split("### Recommendations")[0])
oral_recommendations = clean_text(oral_analysis.split("### Recommendations")[1])

# Store extracted data into sections dictionary
sections['oral_findings'] = oral_findings
sections['oral_conditions'] = oral_conditions
sections['oral_recommendations'] = oral_recommendations

# Print final structured dictionary
print(sections)
