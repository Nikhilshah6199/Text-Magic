
import sys

def get_accent_message():
    """
    Returns:
      A list of accents
    """
    
    accent = ['Australian', 'South Africa', 'British',
            'Indian', 'Canadian', 'Irish', 'Spanish']
    return accent



def get_accent_tld(user_input):
    """
    It takes a user input, maps it to a number, and then returns the corresponding top level domain
    
    Args:
      user_input: The user input from the Web Ui
    
    Returns:
      The accent code
    """

    accent_input = {
        'Australian': 1,
        'South Africa': 2,
        'British': 3,
        'Indian': 4,
        'Canadian': 5,
        'Irish': 6,
        'Spanish': 7
    }
    number = accent_input.get(user_input)
    # Map the input
    accent_map = {
        1: 'com.au',
        2: 'co.za',
        3: 'co.uk',
        4: 'co.in',
        5: 'ca',
        6: 'ie',
        7: 'es'
    }
    # Write function to return accent code
    if number in accent_map:
        return accent_map[number]
    else:
        pass

