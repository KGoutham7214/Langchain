import { useRef, useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const YOU = "you";
const AI = "ai";
function App() {

  const inputRef = useRef();
  const [qna, setQna] = useState([
    {from:YOU , value:"Hi Bot"},{from:AI , value:"hi Human"}
  ]);
  const [loading, setLoading] = useState(false);

  const updateQNA = (from, value) => {
    setQna((qna) => [...qna, { from, value }]);
  };
  useEffect(() => {
    const keyDownHandler = event => {
      console.log('User pressed: ', event.key);

      if (event.key === 'Enter') {
        event.preventDefault();

        // 👇️ call submit function here
        handleSend();
      }
    };

    document.addEventListener('keydown', keyDownHandler);

    return () => {
      document.removeEventListener('keydown', keyDownHandler);
    };
  }, []);

  const handleSend = async() => {
    const question = inputRef.current.value;
    updateQNA(YOU, question);

    setLoading(true);
    await axios
      .post("http://localhost:3000/", {
        question,
      })
      .then((response) => {
        updateQNA(AI, response.data.answer);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  const renderContent = (qna) => {
    const value = qna.value;

    if (Array.isArray(value)) {
      return value.map((v) => <p className="message-text">{v}</p>);
    }

    return <p className="message-text">{value}</p>;
  };
  return (
    <main class="container">
      <div class="chats">
        {qna.map((qna) => {
          if (qna.from === YOU) {
            return (
              <div class="send chat">
                <img
                  src="https://cdn-icons-png.flaticon.com/512/2202/2202112.png"
                  alt=""
                  class="avtar"
                />
                <p>{renderContent(qna)}</p>
              </div>
            );
          }
          return (
            <div class="recieve chat">
              <img
                src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png"
                alt=""
                class="avtar"
              />
              <p>{renderContent(qna)}</p>
            </div>
          );
        })}

        {loading && (
          <div class="recieve chat">
            <img
              src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png"
              alt=""
              class="avtar"
            />
            <p>Typing...</p>
          </div>
        )}
      </div>

      <div class="chat-input">
        <input
          type="text"
          ref={inputRef}
          class="form-control col"
          placeholder="Type Something"
        />
        <button disabled={loading} class="btn btn-success" onClick={handleSend}>
          Send
        </button>
      </div>
    </main>
  );
}

export default App;