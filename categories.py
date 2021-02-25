from collections import Counter

CATEGORIES = [
    {
        'name': 'during',
        'options': [
            'during'
        ]
    },
    {
        'name': 'how is/are',
        'options': [
            'hoe does ',
            'hoe did ',
            'how are ',
            'how can ',
            'how could ',
            'how did ',
            'how do ',
            'how does ',
            'how had',
            'how has ',
            'how have ',
            'how is ',
            'how might ',
            'how was ',
            'how were ',
            'how will',
            'how would',
            'how?',
        ]
    },
    {
        'name': 'how big/size',
        'options': [
            'how big',
            'how common',
            'how deep',
            'how early',
            'how far',
            'how high',
            'how large',
            'how long',
            'how often',
            'how soon',
            'how well',
            'how wide',
            'how ',
            'hoe ' #typo
        ]
    },
    {
        'name': 'how many/much',
        'options': [
            'how many',
            'how much',
            'ho many ', #typo
            'how man ', #typo
            'howmany', #typo
            'howmuch', #typo
            'hwo many', #typo
            'hwo much', #typo
            'of how may', #typo
            'amount',
            'number'
        ]
    },
    {
        'name': 'how old',
        'options': [
            'age',
            'how old',
            'hw old' #typo
        ]
    },
    {
        'name': 'what',
        'options': [
            '.$', # ends in a statement
            'as$',
            'by$',
            'is$',
            'of$',
            'as?$',
            'by?$',
            'is?$',
            'of?$',
            'what',
            'waht ', #typo
            'wat ', #typo
            'whar ', #typo
            'whad ', #typo
            'whas ', #typo
            'whay ', #typo'
            'whhat ', #typo
            'wwhat ', #typo
            '^wha ', #typo
            'which',
            'whic ', #typo
            'wich', #typo
            'whcich', #typo
        ]
    },
    {
        'name': 'when',
        'options': [
            '^date',
            'date ',
            'date?',
            'what year',
            'hat years', #typo
            'what month',
            'what date',
            'what time',
            'waht time', #typo
            'date',
            'month',
            'time'
            'year',
            'when',
            'whaen', #typo
            'whn ', #typo
            'whne ', #typo
        ]
    },
    {
        'name': 'where',
        'options': [
            'where',
            'wher ',
            'wher e '
        ]
    },
    {
        'name': 'who/whom',
        'options': [
            'who',
            'whom',
            'whio', #typo
            'wwho ', #typo
        ]
    },
    {
        'name': 'why',
        'options': [
            'why',
            '^whi ', #typo
        ]
    },
    {
        'name': 'instruction',
        'options': [
            '^define ',
            '^identify ',
            '^name a ',
            '^name some ',
            '^name the ',
            '^name two ',
            '^name ',
            '^nam ', #typo
            '^list '
        ]
    },
    {
        'name': 'yes/no',
        'options': [
            '^are ',
            '^can',
            '^could',
            '^did ',
            '^do ',
            '^does ',
            '^has ',
            '^have ',
            '^is ',
            '^was ',
            '^wa ', #typo
            '^were ',
            '^will ',
            '^would ',
        ]
    },
    {
        'name': 'undefined',
        'options': []
    }
]

def getCategoryIds(question: str):

    NUM_CATS = len(CATEGORIES)

    # Convert to lower case
    q = question.lower()
    
    # Substring replacements
    q = q.replace('\u200b', '').replace(',', ', ').replace('  ', ' ')
    
    if not q[0].isalnum():
        q = q[1:]
      
    # Strip whitespaces
    q = q.strip()
    
    categoryIds = []
    
    # Flag to determine whether to categorize as 'undefined'
    added = False

    for id, cat in enumerate(CATEGORIES):
        for o in cat['options']:
            if o.startswith('^'):
                o = o[1:]
                
                if o.endswith('$'):
                    o = o[:-1]
                    if q == o.lower():
                        categoryIds.append(id)
                        added = True
                        break
                
                elif q.startswith(o.lower()) or (', ' + o.lower()) in q:
                    categoryIds.append(id)
                    added = True
                    break
                    
            elif o.endswith('$'):
                o = o[:-1]
                if q.endswith(o.lower()):
                    categoryIds.append(id)
                    added = True
                    break

            elif q.startswith(o.lower()) or (' ' + o.lower()) in q:
                categoryIds.append(id)
                added = True
                break
    
    if not added:
        #'undefined' category
        categoryIds.append(NUM_CATS-1)
    
    return categoryIds

def printStatistics(questions):

    if len(questions) == 0:
        print('Total: 0 questions.')
        return

    ctr = Counter()
    duplicate_count = 0

    for q in questions:
        categoryIds = getCategoryIds(q)

        assert len(categoryIds) > 0

        duplicate_count += len(categoryIds)  - 1

        for id in categoryIds:
            ctr[id] += 1

    total = 0
    for id, cat in enumerate(CATEGORIES):
        n = ctr[id]
        percentage = n / len(questions) * 100
        total += n
        print(cat['name'], n, f'({percentage:.2f}%)')

    percentage = duplicate_count / len(questions) * 100
    print('Duplicates (Excluded from total)', duplicate_count, f'({percentage:.2f}%)')

    total -= duplicate_count
    percentage = total / len(questions) * 100
    print('Total (Excluding from dupes)', total, f'({percentage:.2f}%)')
