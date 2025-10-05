"use client";
import Engine from './Engine'

const Entry: React.FC = () => {

  return (
    <div>
      <Engine />
      <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
        <h1>Lorem Ipsum Scroll Content</h1>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
        <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>

        {[...Array(10)].map((_, i) => (
          <div key={i}>
            <h2>Section {i + 1}</h2>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce euismod, nunc sit amet aliquam lacinia, nisi enim lobortis enim, vel lacinia nunc enim eget nunc.</p>
            <p>Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Vestibulum tortor quam, feugiat vitae, ultricies eget, tempor sit amet, ante.</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Entry;

