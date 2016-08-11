scp -r -i Home.pem deep-sa/preprocessed_data_filtered_cleaned_8M/ ubuntu@ec2-52-59-39-2.eu-central-1.compute.amazonaws.com:deep-sa/
scp -r -i Home.pem deep-sa/preprocessed_data_filtered_cleaned_16M/ ubuntu@ec2-52-59-39-2.eu-central-1.compute.amazonaws.com:deep-sa/
scp -r -i Home.pem deep-sa/preprocessed_data_filtered_cleaned_40M/ ubuntu@ec2-52-59-39-2.eu-central-1.compute.amazonaws.com:deep-sa/
scp -i Home.pem deep-sa/semeval/filtered_cleaned_8M.gz ubuntu@ec2-52-59-39-2.eu-central-1.compute.amazonaws.com:deep-sa/semeval
scp -i Home.pem deep-sa/semeval/filtered_cleaned_16M.gz ubuntu@ec2-52-59-39-2.eu-central-1.compute.amazonaws.com:deep-sa/semeval
scp -i Home.pem deep-sa/semeval/filtered_cleaned_40M.gz ubuntu@ec2-52-59-39-2.eu-central-1.compute.amazonaws.com:deep-sa/semeval