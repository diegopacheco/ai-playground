pub fn drums_prompt(round: usize, context: &str) -> String {
    format!(
        "You are a drummer in a band composing a song. Round {} of collaboration.\n\
        Previous parts from other musicians:\n{}\n\n\
        Write a drum pattern in ABC notation. Use channel voice 'Perc' clef=perc.\n\
        Keep it 8-16 bars. Use z for rests, letters for hits.\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        Make it groovy and complement the other parts.",
        round, context
    )
}

pub fn bass_prompt(round: usize, context: &str) -> String {
    format!(
        "You are a bass player in a band composing a song. Round {} of collaboration.\n\
        Previous parts from other musicians:\n{}\n\n\
        Write a bass line in ABC notation.\n\
        Use bass clef, keep it 8-16 bars. Use low octave notes (C, D, E, etc).\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        Make it funky and lock in with the drums.",
        round, context
    )
}

pub fn melody_prompt(round: usize, context: &str) -> String {
    format!(
        "You are a melody/lead player in a band composing a song. Round {} of collaboration.\n\
        Previous parts from other musicians:\n{}\n\n\
        Write a melody line in ABC notation.\n\
        Use treble clef, keep it 8-16 bars. Make it singable and catchy.\n\
        Output ONLY the ABC notation block starting with X: and nothing else.\n\
        Complement the bass and drums already written.",
        round, context
    )
}

pub fn lyrics_prompt(round: usize, context: &str) -> String {
    format!(
        "You are a lyricist in a band composing a song. Round {} of collaboration.\n\
        The melody and music so far:\n{}\n\n\
        Write lyrics that fit the melody. Use ABC notation 'w:' lines for words.\n\
        Output ONLY the lyrics as w: lines that can be appended to the melody ABC notation.\n\
        Keep it 4-8 lines. Make it creative and match the musical mood.",
        round, context
    )
}
