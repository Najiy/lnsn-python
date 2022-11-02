test = ["ABC*", "ABC**"]

print("ABC" in [x.replace("*", "") for x in test])