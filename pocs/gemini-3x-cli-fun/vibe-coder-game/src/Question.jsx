import React from 'react';

const Question = ({ question, handleAnswer }) => {
  return (
    <div>
      <h2>{question.question}</h2>
      <div>
        {question.options.map((option, index) => (
          <button
            key={index}
            onClick={() => handleAnswer(index === question.correctAnswer)}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
};

export default Question;
