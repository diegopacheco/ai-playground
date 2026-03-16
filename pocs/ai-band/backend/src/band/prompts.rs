pub fn drums_prompt(round: usize, context: &str) -> String {
    format!(
        "You are a drummer in a band composing a song. Round {} of collaboration.\n\
        Previous parts from other musicians:\n{}\n\n\
        Write a drum pattern in ABC notation. Use channel voice 'Perc' clef=perc.\n\
        IMPORTANT: Use %%MIDI channel 10 for drums.\n\
        Keep it 8-16 bars. Use z for rests, letters for hits.\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        The X: field MUST be X:1\n\
        Make it groovy and complement the other parts.",
        round, context
    )
}

pub fn bass_prompt(round: usize, context: &str) -> String {
    format!(
        "You are a bass player in a band composing a song. Round {} of collaboration.\n\
        Previous parts from other musicians:\n{}\n\n\
        Write a bass line in ABC notation.\n\
        IMPORTANT: Use %%MIDI channel 1 and %%MIDI program 33 for bass.\n\
        Use bass clef, keep it 8-16 bars. Use low octave notes (C, D, E, etc).\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        The X: field MUST be X:1\n\
        Make it funky and lock in with the drums.",
        round, context
    )
}

pub fn melody_prompt(round: usize, context: &str) -> String {
    format!(
        "You are a melody/lead player in a band composing a song. Round {} of collaboration.\n\
        Previous parts from other musicians:\n{}\n\n\
        Write a melody line in ABC notation.\n\
        IMPORTANT: Use %%MIDI channel 1 and %%MIDI program 1 for piano melody.\n\
        Use treble clef, keep it 8-16 bars. Make it singable and catchy.\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        The X: field MUST be X:1\n\
        Complement the bass and drums already written.",
        round, context
    )
}

pub fn lyrics_prompt(round: usize, context: &str, lyrics_theme: &str) -> String {
    let theme_line = if lyrics_theme.is_empty() {
        String::new()
    } else {
        format!("\nIMPORTANT: The lyrics MUST be about or influenced by this theme: {}\n", lyrics_theme)
    };
    format!(
        "You are a lyricist in a band composing a song. Round {} of collaboration.\n\
        The melody and music so far:\n{}\n{}\n\
        Write lyrics that fit the melody. Use ABC notation 'w:' lines for words.\n\
        Output ONLY the lyrics as w: lines that can be appended to the melody ABC notation.\n\
        Keep it 4-8 lines. Make it creative and match the musical mood.",
        round, context, theme_line
    )
}

pub fn singer_prompt(context: &str) -> String {
    format!(
        "You are a singer performing the final version of a collaboratively composed song.\n\
        Here is the full song with all parts:\n{}\n\n\
        Write a vocal part in ABC notation that a singer would perform.\n\
        IMPORTANT: Use %%MIDI channel 1 and %%MIDI program 54 for voice.\n\
        Include the w: lyric lines under the vocal melody notes.\n\
        The vocal melody should follow the main melody but be slightly simplified for singing.\n\
        Use treble clef. The X: field MUST be X:1\n\
        Output ONLY the ABC notation block starting with X: and nothing else.",
        context
    )
}
